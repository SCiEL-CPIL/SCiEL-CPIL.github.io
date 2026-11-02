import math
import numpy as np
import time
import random
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # 1. Generate Initial Valid Configuration (Honeycomb Lattice)
    # We need 11 hexagons. A 3x3 grid of hexagons has 9. Add 2 more.
    # Axial coords for a compact cluster:
    # Center (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: We need 4 more to make 11.
    
    # Let's use a known good packing for 11.
    # 1 center, 6 around it, 4 more attached to the "sides".
    # Axial coordinates:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2) -- these 4 extend the shape.
    
    init_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # State: list of [x, y, angle]
    state = []
    for q, r in init_axial:
        x, y = axial_to_cartesian(q, r)
        state.append([x, y, 0.0])
        
    state = np.array(state)
    
    # Outer hexagon state: [cx, cy, angle]
    # Initial outer hexagon: centered at mean, angle 0
    cx = np.mean(state[:, 0])
    cy = np.mean(state[:, 1])
    phi = 0.0
    
    # Function to get vertices of a hexagon
    def get_hex_vertices(x, y, angle):
        # Vertices of a unit hexagon centered at (0,0) with angle 0
        # Side 1. Vertices are at distance 1 from center.
        # Angles: 0, 60, 120, 180, 240, 300
        verts = []
        for k in range(6):
            a = angle + k * math.pi / 3.0
            verts.append((x + math.cos(a), y + math.sin(a)))
        return np.array(verts)

    # Function to check if two polygons (hexagons) overlap
    # Using Separating Axis Theorem (SAT) for convex polygons.
    # For two convex polygons, they are disjoint if there exists an axis where their projections do not overlap.
    # The axes to check are the normals to the edges of both polygons.
    def polygons_disjoint(v1, v2):
        # v1, v2 are (6,2) arrays
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                # Normal to edge
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    n = n / norm
                    axes.append(n)
        
        for n in axes:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 or max2 < min1:
                return True, 0.0 # Disjoint, gap = 0
            
            # Calculate overlap depth if not disjoint on this axis
            # We want the minimum overlap across all axes to be > 0 for disjointness?
            # No, if ANY axis is disjoint, they are disjoint.
            # If NO axis is disjoint, they overlap.
            # We need a measure of "how much" they overlap for the penalty.
            # Let's return False and a penalty value later.
            
        return False, 0.0

    # To get a smooth penalty, we can calculate the maximum penetration depth.
    # If they are disjoint, penalty is 0.
    # If they overlap, penalty is the maximum "gap" we would need to create?
    # Simpler: If they overlap, penalty = (Overlap Depth)^2.
    # Overlap depth on an axis = min(max1, max2) - max(min1, min2).
    # If this is positive for all axes, they overlap.
    # The "distance" to separate them is the minimum of these positive overlaps? No, it's more complex.
    # Let's just use a binary penalty + small gradient hint if possible.
    # Actually, for SA, a simple large penalty if overlapping is often enough if steps are small.
    
    def calc_penalty_and_size(state, cx, cy, phi):
        # Calculate vertices for all hexagons
        verts = []
        all_points = []
        for i in range(N):
            x, y, ang = state[i]
            v = get_hex_vertices(x, y, ang)
            verts.append(v)
            all_points.append(v)
        
        all_points = np.vstack(all_points) # (66, 2)
        
        # Calculate Outer Hexagon Size
        # Normals for outer hexagon
        normals = []
        for k in range(6):
            ang = phi + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        normals = np.array(normals)
        
        # Project all points onto normals
        # relative to center
        rel_points = all_points - np.array([cx, cy])
        projs = rel_points @ normals.T # (66, 6)
        
        # Max absolute projection
        max_abs_proj = np.max(np.abs(projs))
        
        # Apothem of outer hexagon must be >= max_abs_proj
        A_out = max_abs_proj
        S = A_out / APOTHEM
        
        # Calculate Overlap Penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                disjoint, _ = polygons_disjoint(verts[i], verts[j])
                if not disjoint:
                    # Calculate penetration depth for penalty
                    # Find the axis with minimum overlap
                    v1, v2 = verts[i], verts[j]
                    axes = []
                    for v in [v1, v2]:
                        for k in range(6):
                            p1 = v[k]
                            p2 = v[(k + 1) % 6]
                            edge = p2 - p1
                            n = np.array([-edge[1], edge[0]])
                            norm = np.linalg.norm(n)
                            if norm > 1e-9:
                                axes.append(n / norm)
                    
                    min_overlap = float('inf')
                    for n in axes:
                        proj1 = np.dot(v1, n)
                        proj2 = np.dot(v2, n)
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        overlap = min(max1, max2) - max(min1, min2)
                        if overlap < 0:
                            # Disjoint on this axis, so they are disjoint overall?
                            # Wait, polygons_disjoint returned False, so no axis is disjoint.
                            # So overlap should be >= 0 for all axes.
                            pass
                        if overlap < min_overlap:
                            min_overlap = overlap
                            
                    if min_overlap > 0:
                        penalty += min_overlap ** 2
                    else:
                        # Should not happen if disjoint check failed
                        penalty += 1.0
        
        return S + 100.0 * penalty, S, penalty

    # Simulated Annealing
    best_state = state.copy()
    best_cx, best_cy, best_phi = cx, cy, phi
    best_score, best_S, best_pen = calc_penalty_and_size(best_state, best_cx, best_cy, best_phi)
    
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    steps = 0
    max_steps = 5000
    
    current_state = state.copy()
    current_cx, current_cy, current_phi = cx, cy, phi
    current_score, current_S, current_pen = best_score, best_S, best_pen
    
    rng = np.random.default_rng(42)
    
    while T > T_min and steps < max_steps:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Generate a neighbor
        new_state = current_state.copy()
        new_cx, new_cy, new_phi = current_cx, current_cy, current_phi
        
        move_type = rng.integers(0, 4)
        
        if move_type == 0:
            # Move one hexagon position
            i = rng.integers(0, N)
            dx = rng.normal(0, 0.1)
            dy = rng.normal(0, 0.1)
            new_state[i][0] += dx
            new_state[i][1] += dy
        elif move_type == 1:
            # Rotate one hexagon
            i = rng.integers(0, N)
            dtheta = rng.normal(0, 0.1)
            new_state[i][2] += dtheta
        elif move_type == 2:
            # Move outer hexagon center
            dx = rng.normal(0, 0.1)
            dy = rng.normal(0, 0.1)
            new_cx += dx
            new_cy += dy
        else:
            # Rotate outer hexagon
            dphi = rng.normal(0, 0.1)
            new_phi += dphi
            
        # Evaluate new state
        new_score, new_S, new_pen = calc_penalty_and_size(new_state, new_cx, new_cy, new_phi)
        
        # Metropolis Acceptance
        delta_E = new_score - current_score
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_state = new_state
            current_cx, current_cy, current_phi = new_cx, new_cy, new_phi
            current_score, current_S, current_pen = new_score, new_S, new_pen
            
            if new_score < best_score:
                best_state = current_state.copy()
                best_cx, best_cy, best_phi = current_cx, current_cy, current_phi
                best_score, best_S, best_pen = new_score, new_S, new_pen
                
        T *= alpha
        steps += 1
        
    # Final Polish with Local Optimization
    # Use minimize to refine the best state found by SA
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Flatten state for minimize
        # Variables: x1, y1, th1, ..., xN, yN, thN, cx, cy, phi
        x0 = np.concatenate([best_state.ravel(), [best_cx, best_cy, best_phi]])
        
        def obj(params):
            s = params[:3*N].reshape(N, 3)
            c = params[3*N:3*N+2]
            p = params[3*N+2]
            score, S, pen = calc_penalty_and_size(s, c[0], c[1], p)
            return score
        
        try:
            res = minimize(obj, x0, method='Powell', options={'maxiter': 100, 'ftol': 1e-8})
            if res.fun < best_score:
                best_params = res.x
                best_state = best_params[:3*N].reshape(N, 3)
                best_cx = best_params[3*N]
                best_cy = best_params[3*N+1]
                best_phi = best_params[3*N+2]
                best_score, best_S, best_pen = calc_penalty_and_size(best_state, best_cx, best_cy, best_phi)
        except Exception:
            pass

    # Construct Output
    # Ensure no overlap with a final check and slight shrink if necessary?
    # The task says "disjoint". If our penalty is 0, they are disjoint.
    # We should verify.
    
    # Calculate final side length
    # We used best_S from the last calc, but let's recalculate to be sure
    final_S, _, _ = calc_penalty_and_size(best_state, best_cx, best_cy, best_phi)
    
    # If there is still penalty, the packing is invalid.
    # We should try to return a valid packing.
    # If best_pen > 0, we might need to fall back to a known valid configuration.
    # However, SA usually finds valid packings.
    # Let's add a safety check: if penalty > 0, use the initial lattice which is valid.
    if best_pen > 1e-5:
        # Fall back to initial lattice
        best_state = state.copy()
        best_cx = np.mean(state[:, 0])
        best_cy = np.mean(state[:, 1])
        best_phi = 0.0
        final_S, _, _ = calc_penalty_and_size(best_state, best_cx, best_cy, best_phi)
        
    inner = []
    for x, y, ang in best_state:
        inner.append([float(x), float(y), float(math.degrees(ang))])
        
    center = [float(best_cx), float(best_cy)]
    side = float(final_S)
    rotation = float(math.degrees(best_phi))
    
    return inner, center, side, rotation
