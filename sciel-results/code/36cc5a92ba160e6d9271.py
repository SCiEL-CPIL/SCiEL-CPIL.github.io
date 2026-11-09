"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _containment_penalty_squared(inner_verts, outer_verts):
    """Penalty for inner hexagon not being fully inside outer hexagon. Squared distance."""
    penalty = 0.0
    n_out = len(outer_verts)
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                dist = -cross / edge_len
                penalty += dist * dist
    return penalty

def _sat_overlap_depth_squared(h1, h2):
    """Minimum penetration depth squared between two convex polygons using SAT. Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    if min_depth == float('inf'):
        return 0.0
    return min_depth * min_depth

def _compute_penalty(params, outer_side):
    """Compute all inner hex vertices and return containment + disjointness penalties."""
    # Outer hexagon: center (0,0), rotation 0, side outer_side
    outer_verts = _hex_verts(0.0, 0.0, outer_side, 0.0)
    
    inner_verts_list = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    penalty = 0.0
    for iv in inner_verts_list:
        penalty += _containment_penalty_squared(iv, outer_verts)
        
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth_squared(inner_verts_list[i], inner_verts_list[j])
            penalty += depth
            
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    def build_hexagonal_lattice(nrows, ncols):
        """Build a hexagonal lattice of positions (offset rows)."""
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions
    
    def build_ring_config():
        """6 around center + 5 more in second ring."""
        positions = [(0.0, 0.0)]
        for i in range(6):
            ang = np.radians(60 * i)
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        for i in range(6):
            ang = np.radians(60 * i + 30)
            positions.append((3.464 * np.cos(ang), 3.464 * np.sin(ang)))
        return positions[:12]
    
    # Generate candidate initial configurations
    configs = []
    
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
        
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
        
    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
        
    ring_pos = build_ring_config()
    if len(ring_pos) == N:
        arr = np.array(ring_pos)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    best_score = float('inf')
    best_result = None
    best_params = None
    
    # Phase 1: Find a valid packing at a larger side length
    # Theoretical min ~ 3.464. Try 4.5, 4.2, 4.0.
    test_sides = [4.5, 4.2, 4.0]
    
    for s in test_sides:
        if time.time() > deadline - 30:
            break
        for config in configs:
            if time.time() > deadline - 30:
                break
            init_params = []
            for (x, y) in config:
                init_params.extend([float(x), float(y), 0.0])
            init_params = np.array(init_params)
            
            def objective(p, s=s):
                return _compute_penalty(p, s)
                
            try:
                result = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 3000}
                )
                if result.fun < 1e-6:
                    best_score = s
                    best_result = (result.x, (0.0, 0.0), s, 0.0)
                    best_params = result.x
                    break
            except:
                pass
        if best_result is not None:
            break

    if best_result is None:
        # Fallback: Return a known valid packing (loose)
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0

    # Phase 2: Binary search for minimal side length
    low = 3.464 # Theoretical min
    high = best_score
    
    # We know high is valid.
    # We want to find the smallest valid s.
    
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        mid = (low + high) / 2.0
        if high - low < 1e-5:
            break
            
        found_valid = False
        # Try to find a valid packing at mid
        # Use best_params as initial guess if available
        init_guess = best_params if best_params is not None else np.zeros(3*N)
        
        # Also try original configs if init_guess fails?
        # For speed, just use init_guess.
        
        def objective_mid(p, s=mid):
            return _compute_penalty(p, s)
            
        try:
            result = minimize(
                objective_mid,
                init_guess,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 3000}
            )
            if result.fun < 1e-6:
                found_valid = True
                high = mid
                best_score = mid
                best_result = (result.x, (0.0, 0.0), mid, 0.0)
                best_params = result.x
            else:
                low = mid
        except:
            low = mid

    # Phase 3: Final polish
    if best_result is not None:
        params, center, side, angle = best_result
        
        def objective_final(p, s=side):
            return _compute_penalty(p, s)
            
        try:
            result = minimize(
                objective_final,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun < 1e-8:
                best_result = (result.x, center, side, angle)
        except:
            pass

    if best_result is not None:
        params, center, side, angle = best_result
        inner = []
        for i in range(N):
            x = float(params[3*i])
            y = float(params[3*i+1])
            ang = float(params[3*i+2])
            inner.append([x, y, ang])
        return inner, (float(center[0]), float(center[1])), float(side), float(angle)
    else:
        # Fallback
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0

# EVOLVE-BLOCK-END
