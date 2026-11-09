import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 12

def _get_lattice_points(pattern):
    """
    Generate center coordinates for a given row length pattern.
    pattern: list of integers, e.g., [3, 4, 3, 2]
    Returns: np.array of shape (N, 2)
    """
    points = []
    sqrt3 = np.sqrt(3)
    # We generate points row by row.
    # To make the shape "centered", we can adjust the x-offset.
    # Standard lattice:
    # Row j, index i in row:
    # x = i * sqrt3 + offset_j
    # y = j * 1.5
    
    # We want the rows to be centered relative to each other to form compact shapes.
    # Let's compute the total width of the "ideal" hexagon to center the rows.
    # Or simpler: just generate them and let the optimizer find the center.
    # However, the relative x-shift between rows must be consistent with the lattice.
    # Row 0: even parity. x = i * sqrt3
    # Row 1: odd parity. x = (i + 0.5) * sqrt3
    # Row 2: even parity. x = i * sqrt3
    # To center them, we should probably shift the whole thing.
    # But since the outer hexagon center is optimized, the absolute position doesn't matter.
    # Only the relative positions do.
    
    for j, row_len in enumerate(pattern):
        if row_len == 0:
            continue
        for i in range(row_len):
            if j % 2 == 0:
                x = i * sqrt3
            else:
                x = (i + 0.5) * sqrt3
            y = j * 1.5
            points.append([x, y])
            
    if len(points) != N:
        return None
        
    return np.array(points)

def _check_connectivity(points, pattern):
    """
    Check if the hexagons defined by the pattern are connected.
    We map points to grid indices (i, j).
    """
    # Reconstruct grid indices
    # This is tricky if we just have points.
    # Better to check connectivity during generation or by checking distances.
    # Two hexagons are connected if their centers are distance sqrt(3).
    # We can build a graph based on distances.
    
    n = len(points)
    if n == 0: return False
    
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            if np.isclose(dist, np.sqrt(3), atol=1e-6):
                adj[i].append(j)
                adj[j].append(i)
    
    # BFS/DFS to check connectivity
    visited = set()
    stack = [0]
    visited.add(0)
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)
                
    return len(visited) == n

def _min_enclosing_hexagon_side(points):
    """
    Find the minimal side length of a regular hexagon containing the given points.
    Returns: (side_length, center_x, center_y, angle_deg)
    """
    # The effective hexagon must contain the centers.
    # Let A be the inradius of the effective hexagon.
    # Side length L_eff = 2 * A / sqrt(3).
    
    def objective(theta_deg):
        theta = np.radians(theta_deg)
        # Normals to the 3 axes of the hexagon
        # A regular hexagon has 6 sides. Normals are at theta, theta+60, ...
        # We only need 3 axes (pairs of opposite sides).
        normals = []
        for k in range(3):
            angle = theta + k * np.pi / 3
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        # Projections of points onto normals
        # P is (N, 2), normals is (3, 2)
        # proj is (N, 3)
        proj = points @ normals.T
        
        # For each axis k, we need min and max
        # The width w_k = max(proj[:, k]) - min(proj[:, k])
        # The constraint is that the center c must satisfy:
        # | (c - p_i) dot n_k | <= A
        # This is equivalent to:
        # c dot n_k - A <= p_i dot n_k
        # -c dot n_k - A <= -p_i dot n_k
        
        # We solve LP for c and A.
        # Variables: x = [cx, cy, A]
        # Minimize A (index 2)
        
        c = np.zeros(3)
        c[2] = 1.0 # Minimize A
        
        A_ub = []
        b_ub = []
        
        for k in range(3):
            n_k = normals[k]
            # n_k dot c - A <= p_i dot n_k
            # n_k[0]*cx + n_k[1]*cy - A <= proj[i, k]
            row1 = np.array([n_k[0], n_k[1], -1.0])
            
            # -n_k dot c - A <= -p_i dot n_k
            # -n_k[0]*cx - n_k[1]*cy - A <= -proj[i, k]
            row2 = np.array([-n_k[0], -n_k[1], -1.0])
            
            for i in range(len(points)):
                A_ub.append(row1)
                b_ub.append(proj[i, k])
                A_ub.append(row2)
                b_ub.append(-proj[i, k])
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x[2] # Return A
            else:
                return 1e9
        except Exception:
            return 1e9

    # Optimize theta in [0, 60)
    # The function is periodic with 60 deg.
    # We can use minimize_scalar.
    res = minimize_scalar(objective, bounds=(0, 60), method='bounded', options={'xatol': 1e-6})
    
    if res.success:
        A_min = res.fun
        theta_opt = res.x
        # Recalculate center for reporting
        # Re-run LP to get cx, cy
        theta = np.radians(theta_opt)
        normals = []
        for k in range(3):
            angle = theta + k * np.pi / 3
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        proj = points @ normals.T
        
        c = np.zeros(3)
        c[2] = 1.0
        A_ub = []
        b_ub = []
        for k in range(3):
            n_k = normals[k]
            row1 = np.array([n_k[0], n_k[1], -1.0])
            row2 = np.array([-n_k[0], -n_k[1], -1.0])
            for i in range(len(points)):
                A_ub.append(row1)
                b_ub.append(proj[i, k])
                A_ub.append(row2)
                b_ub.append(-proj[i, k])
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res_lp.success:
            cx = res_lp.x[0]
            cy = res_lp.x[1]
            A = res_lp.x[2]
            L_eff = 2 * A / np.sqrt(3)
            L = L_eff + 1.0
            return L, cx, cy, theta_opt
        else:
            return 1e9, 0, 0, 0
    else:
        return 1e9, 0, 0, 0

def _generate_patterns():
    """Generate all compositions of 12 into at most 6 parts."""
    patterns = []
    def backtrack(remaining, max_part, current):
        if remaining == 0:
            patterns.append(list(current))
            return
        if len(current) >= 6:
            return
        for i in range(min(remaining, max_part), 0, -1):
            current.append(i)
            backtrack(remaining - i, i, current)
            current.pop()
            
    backtrack(12, 12, [])
    return patterns

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_L = float('inf')
    best_inner = None
    best_center = None
    best_angle = 0.0
    
    patterns = _generate_patterns()
    
    # Shuffle patterns to ensure diverse search if time runs out?
    # Or just iterate. There are ~2000 patterns.
    
    for pattern in patterns:
        if time.time() > deadline:
            break
            
        points = _get_lattice_points(pattern)
        if points is None:
            continue
            
        if not _check_connectivity(points, pattern):
            continue
            
        L, cx, cy, angle = _min_enclosing_hexagon_side(points)
        
        if L < best_L:
            best_L = L
            best_inner = [(p[0], p[1], 0.0) for p in points]
            best_center = (cx, cy)
            best_angle = angle
    
    if best_inner is None:
        # Fallback
        inner = [(4.0 * k, 0.0, 0.0) for k in range(-6, 6)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_inner, best_center, best_L, best_angle

