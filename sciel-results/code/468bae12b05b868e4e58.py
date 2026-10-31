"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math
from itertools import combinations

def _hex_vertices_array(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon as (6,2) array."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _all_vertices(positions, angles, side=1.0):
    """Get all vertices of all hexagons as (N*6, 2) array."""
    n = len(positions)
    verts = np.zeros((n * 6, 2))
    for i in range(n):
        verts[i*6:(i+1)*6] = _hex_vertices_array(positions[i][0], positions[i][1], side, angles[i])
    return verts

def _min_enclosing_hex_side(positions, angles, outer_center=(0.0, 0.0), outer_rotation_deg=0.0):
    """Compute minimum side length of outer hexagon centered at outer_center with given rotation
    to contain all inner hexagons."""
    verts = _all_vertices(positions, angles, 1.0)
    verts = verts - np.array(outer_center)
    
    # For a hexagon with side S, rotation phi, centered at origin:
    # A point (x,y) is inside iff for all k=0..5:
    #   x*cos(theta_k) + y*sin(theta_k) <= S * sqrt(3)/2
    # where theta_k = phi + (2k+1)*pi/6
    phi = np.radians(outer_rotation_deg)
    thetas = phi + np.array([(2*k+1)*np.pi/6 for k in range(6)])
    
    max_proj = 0.0
    for theta in thetas:
        proj = verts[:, 0] * np.cos(theta) + verts[:, 1] * np.sin(theta)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    
    return max_proj / (np.sqrt(3) / 2)

def _check_overlap_pair(p1, a1, p2, a2, side=1.0):
    """Check if two unit hexagons overlap using SAT (Separating Axis Theorem).
    p1, p2 are (x,y) tuples, a1, a2 are angles in degrees."""
    v1 = _hex_vertices_array(p1[0], p1[1], side, a1)
    v2 = _hex_vertices_array(p2[0], p2[1], side, a2)
    
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    axes = []
    for e in list(edges1) + list(edges2):
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            axes.append(n / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
            return False
    
    return True

def _check_all_disjoint(positions, angles):
    """Check all pairs are disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if _check_overlap_pair(positions[i], angles[i], positions[j], angles[j]):
                return False
    return True

def _logsumexp_penalty(positions, angles, beta=100.0):
    """Compute a smooth approximation of the overlap constraint using log-sum-exp.
    Returns a penalty that is ~0 when no overlap, large when overlap exists."""
    n = len(positions)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # Compute minimum distance between the two hexagons
            v1 = _hex_vertices_array(positions[i][0], positions[i][1], 1.0, angles[i])
            v2 = _hex_vertices_array(positions[j][0], positions[j][1], 1.0, angles[j])
            
            # Use SAT to get the minimum gap (negative means overlap)
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            
            min_gap = float('inf')
            for e in list(edges1) + list(edges2):
                n_vec = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n_vec)
                if norm < 1e-12:
                    continue
                n_vec = n_vec / norm
                
                proj1 = np.dot(v1, n_vec)
                proj2 = np.dot(v2, n_vec)
                
                # Gap in this direction
                gap = min(np.max(proj1) - np.min(proj2), np.max(proj2) - np.min(proj1))
                if gap < min_gap:
                    min_gap = gap
            
            # If min_gap < 0, there's overlap. Use log-sum-exp to smooth.
            # We want penalty to be large when min_gap is very negative.
            # Use exp(-beta * min_gap) which is 1 when min_gap=0, large when min_gap<0
            # Actually, let's use a sigmoid-like penalty
            if min_gap < 0:
                penalty += math.exp(-beta * min_gap)
            else:
                penalty += 0.0
    
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    sqrt3 = math.sqrt(3)
    
    # Generate candidate arrangements using hexagonal grid
    def axial_to_xy(q, r):
        return (1.5 * q, sqrt3 * (r + q / 2.0))
    
    arrangements = []
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    
    # Option 1: 4 consecutive from ring2
    for start in range(12):
        subset = [ring2[(start + k) % 12] for k in range(4)]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Option 2: Symmetric selections (opposite pairs)
    for i in range(6):
        for j in range(i+1, 6):
            subset = [ring2[i], ring2[i+6], ring2[j], ring2[j+6]]
            axial_coords = [(0,0)] + ring1 + subset
            positions = [axial_to_xy(q, r) for q, r in axial_coords]
            arrangements.append(positions)
    
    # Also try other compact arrangements
    # 3-4-3-1 row arrangement
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon
    
    # Generate more diverse arrangements
    # Try shifting rows to create different topologies
    
    # Arrangement: 2-5-2-2 (or similar)
    # Let's try a few more patterns
    
    # Pattern: center + 6 ring1 + 4 ring2 but with different selections
    # We already have this above. Let's also try:
    
    # 1 + 6 + 4 where the 4 are not consecutive
    for combo in combinations(range(12), 4):
        subset = [ring2[i] for i in combo]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        arrangements.append(positions)
    
    # Deduplicate arrangements
    unique_arrangements = []
    seen = set()
    for arr in arrangements:
        key = tuple((round(p[0], 6), round(p[1], 6)) for p in arr)
        if key not in seen:
            seen.add(key)
            unique_arrangements.append(arr)
    
    arrangements = unique_arrangements
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_rotation = 0.0
    
    # First pass: evaluate all arrangements with outer rotation 0 and 30 degrees
    for arr in arrangements:
        if time.time() > deadline - 30.0:
            break
        angles = [0.0] * N
        if not _check_all_disjoint(arr, angles):
            continue
        
        # Try outer rotation 0 and 30
        for outer_rot in [0.0, 30.0]:
            side = _min_enclosing_hex_side(arr, angles, (0.0, 0.0), outer_rot)
            if side < best_side:
                best_side = side
                best_positions = arr[:]
                best_angles = angles[:]
                best_outer_rotation = outer_rot
    
    if best_positions is None:
        # Fallback: linear arrangement
        best_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        best_angles = [0.0] * N
        best_side = _min_enclosing_hex_side(best_positions, best_angles)
        best_outer_rotation = 0.0
    
    # Second pass: optimize positions using gradient-based method
    # Use the log-sum-exp penalty to create a smooth objective
    
    def objective(params):
        """Objective function for optimization.
        params: flattened array of [x0, y0, x1, y1, ..., xN-1, yN-1]
        Returns: outer side length + penalty for overlaps"""
        positions = [(params[2*i], params[2*i+1]) for i in range(N)]
        angles = [0.0] * N
        
        # Compute outer side
        side = _min_enclosing_hex_side(positions, angles, (0.0, 0.0), best_outer_rotation)
        
        # Compute overlap penalty
        penalty = _logsumexp_penalty(positions, angles, beta=50.0)
        
        return side + penalty * 10.0
    
    # Start from the best arrangement found
    initial_params = np.array([best_positions[i][0], best_positions[i][1]] for i in range(N)).flatten()
    
    # Run optimization with time limit
    opt_start = time.time()
    opt_deadline = opt_start + 90.0  # Leave 20s for final checks
    
    try:
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() < opt_deadline and result.success:
            opt_positions = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
            opt_angles = [0.0] * N
            
            # Verify validity
            if _check_all_disjoint(opt_positions, opt_angles):
                opt_side = _min_enclosing_hex_side(opt_positions, opt_angles, (0.0, 0.0), best_outer_rotation)
                if opt_side < best_side:
                    best_side = opt_side
                    best_positions = opt_positions
                    best_angles = opt_angles
    except Exception:
        pass
    
    # Third pass: try rotating individual hexagons to improve packing
    # For each hexagon, try small rotations and see if it helps
    
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(N):
                for delta_angle in [-15, -10, -5, 5, 10, 15]:
                    test_angles = best_angles[:]
                    test_angles[i] = (test_angles[i] + delta_angle) % 360
                    
                    if _check_all_disjoint(best_positions, test_angles):
                        test_side = _min_enclosing_hex_side(best_positions, test_angles, (0.0, 0.0), best_outer_rotation)
                        if test_side < best_side - 1e-6:
                            best_side = test_side
                            best_angles = test_angles
                            improved = True
    
    # Final check: try different outer rotations
    if time.time() < deadline - 5.0:
        for outer_rot in [0.0, 15.0, 30.0, 45.0, 60.0]:
            side = _min_enclosing_hex_side(best_positions, best_angles, (0.0, 0.0), outer_rot)
            if side < best_side:
                best_side = side
                best_outer_rotation = outer_rot
    
    # Add tiny epsilon for strict containment
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, best_outer_rotation
# EVOLVE-BLOCK-END
