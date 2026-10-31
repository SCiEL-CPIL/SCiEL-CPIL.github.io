"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START

def _build_contact_graph(pos, radii, margin=1.01):
    """
    Build the active contact graph.
    Returns:
        edges: list of (i, j) for circle-circle contacts
        wall_edges: list of (i, wall_id) for boundary contacts
                   wall_id: 0=left, 1=right, 2=bottom, 3=top
    """
    n = len(radii)
    edges = []
    wall_edges = []
    
    # Circle-Circle
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            # If they are close, consider them in contact
            if dist < margin * (radii[i] + radii[j]):
                edges.append((i, j))
                
    # Circle-Wall
    for i in range(n):
        x, y, r = pos[i, 0], pos[i, 1], radii[i]
        # Left wall: x - r = 0
        if abs(x - r) < margin * r * 0.5:
            wall_edges.append((i, 0))
        # Right wall: (1-x) - r = 0
        if abs((1.0 - x) - r) < margin * r * 0.5:
            wall_edges.append((i, 1))
        # Bottom wall: y - r = 0
        if abs(y - r) < margin * r * 0.5:
            wall_edges.append((i, 2))
        # Top wall: (1-y) - r = 0
        if abs((1.0 - y) - r) < margin * r * 0.5:
            wall_edges.append((i, 3))
            
    return edges, wall_edges

def _solve_lp_for_radii(pos, edges, wall_edges, deadline):
    """
    Solve LP to maximize sum(radii) given fixed positions and active constraints.
    Variables: r_0, ..., r_{N-1}
    Objective: max sum(r) -> min -sum(r)
    
    Constraints:
    1. For each edge (i,j): r_i + r_j <= dist_ij
    2. For each wall edge (i, w): r_i <= dist_to_wall_i
    3. r_i > 0
    """
    n = len(pos)
    c = -np.ones(n) # Minimize -sum(r)
    
    A_ub = []
    b_ub = []
    
    # Circle-Circle constraints: r_i + r_j <= dist_ij
    for i, j in edges:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        row = np.zeros(n)
        row[i] = 1.0
        row[j] = 1.0
        A_ub.append(row)
        b_ub.append(dist)
        
    # Wall constraints: r_i <= dist_to_wall
    for i, w in wall_edges:
        x, y = pos[i]
        if w == 0: dist = x
        elif w == 1: dist = 1.0 - x
        elif w == 2: dist = y
        else: dist = 1.0 - y
        
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(dist)
        
    if not A_ub:
        return np.full(n, 0.01)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(1e-6, None) for _ in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback to simple estimation if LP fails
            radii = np.zeros(n)
            for i in range(n):
                x, y = pos[i]
                r = min(x, 1-x, y, 1-y)
                for j in range(n):
                    if i == j: continue
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    r = min(r, max(d - 0.01, 0.01))
                radii[i] = max(r * 0.9, 1e-5)
            return radii
    except Exception:
        radii = np.zeros(n)
        for i in range(n):
            x, y = pos[i]
            r = min(x, 1-x, y, 1-y)
            radii[i] = max(r * 0.9, 1e-5)
        return radii

def _topological_optimize(pos, radii, deadline):
    """
    Iteratively refine positions and radii using the contact graph.
    1. Identify active constraints.
    2. Solve LP for optimal radii.
    3. Adjust positions to tighten constraints (local Newton/Gradient step).
    """
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # Iterative refinement
    for it in range(100):
        if time.time() > deadline - 5:
            break
            
        # 1. Build graph
        edges, wall_edges = _build_contact_graph(pos, radii, margin=1.02)
        
        # 2. Solve LP for radii
        new_radii = _solve_lp_for_radii(pos, edges, wall_edges, deadline)
        
        # Check if this is valid and better
        pos_pp, radii_pp = _post_process(pos, new_radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                radii = radii_pp.copy() # Update working radii
                
        # 3. Local position refinement
        # We want to move circles to increase the "slack" in the tightest constraints
        # or simply perform a local search to find a better geometry for the same topology.
        
        # Simple gradient step: Move circles apart if they are overlapping in the LP solution?
        # Actually, LP ensures no overlap. The issue is that positions might be suboptimal 
        # for the *next* LP solve. 
        # We can try a small perturbation + re-solve.
        
        # Perturb positions slightly
        rng = np.random.RandomState(it * 10 + 1)
        step_size = 0.01 / (1 + it * 0.1)
        
        pos_pert = pos + rng.normal(0, step_size, pos.shape)
        # Clamp to box
        pos_pert[:, 0] = np.clip(pos_pert[:, 0], 0.01, 0.99)
        pos_pert[:, 1] = np.clip(pos_pert[:, 1], 0.01, 0.99)
        
        # Re-solve LP with perturbed positions
        edges_p, wall_edges_p = _build_contact_graph(pos_pert, radii, margin=1.02)
        radii_p = _solve_lp_for_radii(pos_pert, edges_p, wall_edges_p, deadline)
        
        pos_pp, radii_pp = _post_process(pos_pert, radii_p)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                pos = pos_pp.copy()
                radii = radii_pp.copy()
            elif score > np.sum(radii):
                # Accept if it improves current, even if not global best, to explore
                # But be careful not to degrade too much.
                if score > np.sum(radii) * 0.99:
                    pos = pos_pp.copy()
                    radii = radii_pp.copy()
                    
    return best_pos, best_radii

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(100):
        valid = True
        # Boundary
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
                
        # Overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
                
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
                
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(200):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
        
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
            
        if max_r < 0.01:
            for attempt in range(300):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
                    
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
        
    return pos, radii

def _try_radius_expansion(pos, radii, max_iterations=100):
    """
    Greedy expansion: try to increase each radius slightly if it doesn't violate constraints.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iterations):
        improved = False
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            
            # Check constraints
            can_expand = True
            max_expansion = 0.01
            
            # Walls
            max_expansion = min(max_expansion, x - r)
            max_expansion = min(max_expansion, 1.0 - x - r)
            max_expansion = min(max_expansion, y - r)
            max_expansion = min(max_expansion, 1.0 - y - r)
            
            # Circles
            for j in range(N):
                if i == j: continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                space = dist - r - radii[j]
                if space < 0:
                    can_expand = False
                    break
                max_expansion = min(max_expansion, space)
                
            if can_expand and max_expansion > 1e-6:
                radii[i] += max_expansion * 0.5
                improved = True
                
        if not improved:
            break
            
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    pos_cb2, rad_cb2 = _initial_corner_biased(seed=13)
    inits.append((pos_cb2, rad_cb2))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    # Run topological optimization on each init
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        try:
            pos_top, rad_top = _topological_optimize(pos_init, rad_init, deadline)
            pos_pp, radii_pp = _post_process(pos_top, rad_top)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception as e:
            pass
    
    # Final polish: Greedy expansion and local search
    if time.time() < deadline - 5:
        best_pos, best_radii = _try_radius_expansion(best_pos, best_radii, max_iterations=50)
        pos_pp, radii_pp = _post_process(best_pos, best_radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
    # One more topological pass on the best result
    if time.time() < deadline - 2:
        try:
            pos_top, rad_top = _topological_optimize(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_top, rad_top)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
