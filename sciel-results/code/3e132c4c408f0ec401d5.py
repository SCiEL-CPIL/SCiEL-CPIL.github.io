import numpy as np
import time
import scipy.optimize as opt

N = 11
SQRT3 = np.sqrt(3.0)
APOTHEM = SQRT3 / 2.0
EPS = 1e-4
# For same orientation, min center distance is sqrt(3). Add epsilon for safety.
MIN_DIST = SQRT3 + 0.05

def _hexagon_vertices(center, angle_deg=0.0):
    """Returns vertices of a unit hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3.0 + angle_rad
    x = center[0] + np.cos(angles)
    y = center[1] + np.sin(angles)
    return np.column_stack((x, y))

def _outer_hexagon_normals(angle_deg=0.0):
    """Returns normal vectors for the sides of the outer hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3.0 + np.pi / 6.0 + angle_rad
    nx = np.cos(angles)
    ny = np.sin(angles)
    return np.column_stack((nx, ny))

def _calculate_required_side(positions, outer_angle_deg=0.0):
    """
    Calculates the minimum side length of the outer hexagon to contain all inner hexagons.
    Uses the support function property: side_length * sqrt(3) = max_proj + min_proj.
    Since we center the bounding box at the origin for calculation efficiency, we just need max |projection|?
    No, the outer hexagon can be translated. The problem asks for the outer hexagon center too.
    Usually, we can assume the outer hexagon is centered at the centroid or origin for minimal size.
    Actually, the minimal enclosing hexagon of a set of points (with fixed orientation) is found by
    finding the minimal strip width in 3 directions (normals at 120 deg).
    For a regular hexagon, the side length L is related to the width W in the normal direction by W = L * sqrt(3).
    So we need to minimize max( W1, W2, W3 ) / sqrt(3).
    
    Let's compute the projection of all vertices onto the 3 independent normals.
    The width in a direction n is max(v.n) - min(v.n).
    We need to satisfy width_i <= L * sqrt(3) for all 3 independent normal directions.
    So L >= max(width_i) / sqrt(3).
    """
    normals = _outer_hexagon_normals(outer_angle_deg)
    # We only need 3 independent normals (0, 1, 2) as the others are negatives or duplicates.
    # Actually normals[0], normals[2], normals[4] are the 3 independent directions.
    n0 = normals[0]
    n1 = normals[2]
    n2 = normals[4]
    
    max_proj = [0, 0, 0]
    min_proj = [float('inf'), float('inf'), float('inf')]
    
    for i in range(len(positions)):
        verts = _hexagon_vertices(positions[i], 0.0)
        projs0 = verts @ n0
        projs1 = verts @ n1
        projs2 = verts @ n2
        
        max_proj[0] = max(max_proj[0], np.max(projs0))
        min_proj[0] = min(min_proj[0], np.min(projs0))
        
        max_proj[1] = max(max_proj[1], np.max(projs1))
        min_proj[1] = min(min_proj[1], np.min(projs1))
        
        max_proj[2] = max(max_proj[2], np.max(projs2))
        min_proj[2] = min(min_proj[2], np.min(projs2))
        
    widths = [max_proj[i] - min_proj[i] for i in range(3)]
    max_width = max(widths)
    return max_width / SQRT3

def _penalty(positions):
    """
    Smooth penalty for overlapping inner hexagons.
    Uses a smooth approximation of max(0, violation).
    """
    penalty = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < MIN_DIST:
                # Use a smooth step-like function or just squared penalty
                # Squared penalty is standard
                penalty += (MIN_DIST - dist) ** 2
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_side = float('inf')
    best_positions = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # --- Fallback Configuration ---
    # Hexagonal lattice packing is a strong baseline.
    spacing = SQRT3 + 0.05 # Minimum distance
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    fallback_positions = []
    for r in range(rows):
        for c in range(cols):
            if len(fallback_positions) < N:
                x = c * spacing
                y = r * spacing * 0.5 # Hex packing: y step is spacing * sqrt(3)/2 ? No.
                # For regular hexagons side 1, distance between centers in hex lattice:
                # Direction 0: (sqrt(3), 0)
                # Direction 60: (sqrt(3)/2, 1.5)
                # Let's use a simpler square grid first for robust fallback, then optimize.
                # Actually, a hex grid is better.
                # Let's just use a square grid with large spacing for guaranteed validity, then optimize.
                pass

    # Re-generate fallback as square grid with safe spacing
    safe_spacing = SQRT3 + 0.5
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    fallback_positions = []
    for r in range(rows):
        for c in range(cols):
            if len(fallback_positions) < N:
                x = c * safe_spacing
                y = r * safe_spacing
                fallback_positions.append([x, y])
    
    fallback_positions = np.array(fallback_positions[:N])
    centroid = np.mean(fallback_positions, axis=0)
    fallback_positions = fallback_positions - centroid

    best_positions = fallback_positions.copy()
    best_side = _calculate_required_side(best_positions)

    # --- Optimization ---
    
    def objective(vars):
        pos = vars.reshape(N, 2)
        pen = _penalty(pos)
        # Weight the penalty heavily to force feasibility
        # If penalty is small, optimize side length
        # If penalty is large, minimize penalty
        if pen > 1e-6:
            return 1e5 * pen
        side = _calculate_required_side(pos)
        return side

    # Use SLSQP or COBYLA? SLSQP is often better for these.
    # Let's try a few restarts with different initializations.
    
    def run_optimization(init_pos, method='L-BFGS-B'):
        nonlocal best_side, best_positions
        if time.time() > deadline:
            return
        
        try:
            x0 = init_pos.flatten()
            
            # L-BFGS-B is fast but can get stuck.
            # Let's try it first.
            if method == 'L-BFGS-B':
                res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            elif method == 'COBYLA':
                # COBYLA is derivative-free, slower but robust to non-smoothness
                res = opt.minimize(objective, x0, method='COBYLA', options={'maxiter': 500, 'rhobeg': 0.1})
            else:
                return

            if time.time() > deadline:
                return

            pos = res.x.reshape(N, 2)
            pen = _penalty(pos)
            if pen < 1e-4: # Feasible enough
                side = _calculate_required_side(pos)
                if side < best_side:
                    best_side = side
                    best_positions = pos
        except Exception:
            pass

    # 1. Optimize from Fallback
    run_optimization(fallback_positions, 'L-BFGS-B')
    
    # 2. If we have a good solution, try to refine with COBYLA or more L-BFGS-B
    if time.time() < deadline - 20:
        # Perturb best solution
        for k in range(5):
            if time.time() > deadline - 10:
                break
            
            # Small perturbation
            noise = np.random.uniform(-0.1, 0.1, (N, 2))
            init = best_positions + noise
            run_optimization(init, 'L-BFGS-B')
            
            # Larger perturbation with COBYLA to escape local minima
            if time.time() < deadline - 20:
                noise2 = np.random.uniform(-0.5, 0.5, (N, 2))
                init2 = best_positions + noise2
                run_optimization(init2, 'COBYLA')

    # 3. Try a few random starts
    for k in range(3):
        if time.time() > deadline - 10:
            break
        random_pos = np.random.uniform(-5, 5, (N, 2))
        random_pos -= np.mean(random_pos, axis=0)
        run_optimization(random_pos, 'L-BFGS-B')

    # --- Final Validation and Cleanup ---
    
    # Check for strict validity
    valid = True
    n = len(best_positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(best_positions[i] - best_positions[j])
            if dist < MIN_DIST - 1e-5: # Small tolerance for floating point
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to the initial safe grid
        best_positions = fallback_positions
        best_side = _calculate_required_side(best_positions)
    
    # The outer hexagon center can be anything, but (0,0) is fine as we centered the points.
    # The side length is the score.
    # We return angles as 0.0.
    
    return [list(p) + [0.0] for p in best_positions], (0.0, 0.0), best_side, 0.0
