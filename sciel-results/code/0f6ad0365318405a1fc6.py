"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Uses a structural change: direct convex optimization of the outer side length
with tight constraint handling, combined with a multi-start local search that
optimizes both inner positions and outer parameters simultaneously.
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg, tol=1e-9):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        a = np.radians(30 + k * 60)
        nx = np.cos(a)
        ny = np.sin(a)
        if lx * nx + ly * ny > inradius + tol:
            return False
    return True

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons are disjoint using SAT."""
    n1 = len(verts1)
    n2 = len(verts2)
    
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    
    return False

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(outer_angle)
            ca = np.cos(-angle_rad)
            sa = np.sin(-angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            r = np.sqrt(lx*lx + ly*ly)
            if r < 1e-12:
                continue
            
            theta = np.arctan2(ly, lx)
            best_cos = 0.0
            for k in range(6):
                a = np.radians(30 + k * 60)
                d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                d = min(d, 2*np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c
            
            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r * best_cos / np.sqrt(3)
                best_side = max(best_side, required_side)
    
    return best_side

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    cx, cy = outer_center
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], cx, cy, outer_side, outer_angle, tol=tol):
                return False
    
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol):
                return False
    
    return True

def _initial_lattice():
    """Create initial hexagonal lattice arrangement with correct spacing."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)  # Minimum distance for non-overlapping unit hexagons
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    # Add 4 more hexagons in a second ring
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    return positions

def _initial_random():
    """Create a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def _objective(params, outer_angle):
    """Objective function: minimize outer side length with penalty for overlaps."""
    # Unpack parameters: 3 per hexagon (x, y, angle)
    positions = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2] % 360
        positions.append([x, y, a])
    
    # Compute required outer side length
    side = _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    
    # Penalty for overlaps
    all_verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in positions]
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=1e-5):
                # Estimate overlap amount
                ci = np.mean(all_verts[i], axis=0)
                cj = np.mean(all_verts[j], axis=0)
                d = np.linalg.norm(ci - cj)
                overlap_penalty += max(0, 2*np.sqrt(3) - d) ** 2
    
    return side + 100.0 * overlap_penalty

def _gradient_descent(positions, outer_angle, max_iters=500, lr=0.05):
    """Simple gradient descent to optimize positions."""
    positions = [list(p) for p in positions]
    params = np.array([p[0] for p in positions] + [p[1] for p in positions] + [p[2] for p in positions])
    
    best_side = _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    best_params = params.copy()
    
    for it in range(max_iters):
        if _remaining_time() < 15.0:
            break
        
        # Compute numerical gradient
        grad = np.zeros(3*N)
        eps = 1e-4
        base_obj = _objective(params, outer_angle)
        
        for i in range(3*N):
            params_plus = params.copy()
            params_plus[i] += eps
            obj_plus = _objective(params_plus, outer_angle)
            grad[i] = (obj_plus - base_obj) / eps
        
        # Update parameters
        params = params - lr * grad
        for i in range(N):
            params[3*i+2] = params[3*i+2] % 360
        
        # Reconstruct positions
        new_positions = []
        for i in range(N):
            new_positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        
        new_side = _min_enclosing_side(new_positions, (0.0, 0.0), outer_angle)
        
        # Check for overlaps
        all_verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in new_positions]
        has_overlap = False
        for i in range(N):
            for j in range(i+1, N):
                if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=1e-5):
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if not has_overlap and new_side < best_side:
            best_side = new_side
            best_params = params.copy()
            positions = new_positions
        
        # Adaptive learning rate
        if it % 50 == 0:
            lr *= 0.9
    
    # Return best found
    best_positions = []
    for i in range(N):
        best_positions.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    return best_positions, best_side

def _scipy_optimize(positions, outer_angle, max_iters=100):
    """Use scipy's L-BFGS-B optimizer."""
    positions = [list(p) for p in positions]
    x0 = np.array([p[0] for p in positions] + [p[1] for p in positions] + [p[2] for p in positions])
    
    best_side = _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    best_x = x0.copy()
    
    def obj_and_grad(x):
        obj = _objective(x, outer_angle)
        # Numerical gradient
        eps = 1e-4
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (_objective(x_plus, outer_angle) - obj) / eps
        return obj, grad
    
    try:
        result = minimize(
            lambda x: _objective(x, outer_angle),
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'maxfun': 10000}
        )
        
        new_positions = []
        for i in range(N):
            new_positions.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2] % 360])
        
        new_side = _min_enclosing_side(new_positions, (0.0, 0.0), outer_angle)
        
        # Check validity
        all_verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in new_positions]
        has_overlap = False
        for i in range(N):
            for j in range(i+1, N):
                if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=1e-5):
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if not has_overlap and new_side < best_side:
            best_side = new_side
            best_x = result.x
            positions = new_positions
    
    except Exception:
        pass
    
    return positions, best_side

def _local_search(positions, outer_center, outer_angle, max_iters=300):
    """Local search with greedy improvement."""
    positions = [list(p) for p in positions]
    best_positions = [list(p) for p in positions]
    best_side = _min_enclosing_side(positions, outer_center, outer_angle)
    
    step = 0.1
    no_improve = 0
    
    for it in range(max_iters):
        if _remaining_time() < 10.0:
            break
        
        improved = False
        for i in range(N):
            # Try small perturbations
            for dx in [-step, -step/2, step/2, step]:
                for dy in [-step, -step/2, step/2, step]:
                    new_pos = [list(p) for p in positions]
                    new_pos[i][0] += dx
                    new_pos[i][1] += dy
                    
                    new_side = _min_enclosing_side(new_pos, outer_center, outer_angle)
                    
                    # Quick overlap check
                    all_verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in new_pos]
                    has_overlap = False
                    for a in range(N):
                        for b in range(a+1, N):
                            if not _hexagons_disjoint(all_verts[a], all_verts[b], tol=1e-5):
                                has_overlap = True
                                break
                        if has_overlap:
                            break
                    
                    if not has_overlap and new_side < best_side - 1e-7:
                        positions = new_pos
                        best_positions = [list(p) for p in positions]
                        best_side = new_side
                        improved = True
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 5:
                step *= 0.7
                if step < 0.001:
                    break
    
    return best_positions, best_side

def construct():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    outer_angles = [0.0, 30.0, 15.0, 45.0, 10.0, 20.0]
    
    for oa in outer_angles:
        if _remaining_time() < 25.0:
            break
        
        # Multiple initial configurations
        initial_configs = [
            _initial_lattice(),
            _initial_random(),
            _initial_random(),
        ]
        
        for init_pos in initial_configs:
            if _remaining_time() < 15.0:
                break
            
            positions = [list(p) for p in init_pos]
            
            # Gradient descent
            positions, side = _gradient_descent(positions, oa, max_iters=200, lr=0.03)
            
            # Scipy optimization
            positions, side = _scipy_optimize(positions, oa, max_iters=50)
            
            # Local search for fine-tuning
            positions, side = _local_search(positions, (0.0, 0.0), oa, max_iters=200)
            
            # Verify validity
            if _check_validity(positions, (0.0, 0.0), side, oa, tol=1e-5):
                if side < best_score:
                    best_score = side
                    best_result = (positions, (0.0, 0.0), side, oa)
            
            # Try with adjusted outer center
            for dcx in [-0.15, -0.05, 0.0, 0.05, 0.15]:
                for dcy in [-0.15, -0.05, 0.0, 0.05, 0.15]:
                    if _remaining_time() < 10.0:
                        break
                    oc = (dcx, dcy)
                    side2 = _min_enclosing_side(positions, oc, oa)
                    if _check_validity(positions, oc, side2, oa, tol=1e-5):
                        if side2 < best_score:
                            best_score = side2
                            best_result = (positions, oc, side2, oa)
    
    if best_result is not None:
        return best_result
    
    # Fallback: known valid configuration
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 3.5
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = 6.0
    for k in range(4):
        a = np.radians(30 + 60 * k)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0
