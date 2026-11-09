import numpy as np
import time
import random
from scipy.optimize import linprog, minimize

def _is_feasible(centers, sides, tol=1e-7):
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def _penalty_objective(x, n=10, penalty_weight=1000.0):
    """
    Smooth penalty objective for SLSQP.
    x = [x_0, y_0, s_0, x_1, y_1, s_1, ...]
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Objective: maximize sum(sides) => minimize -sum(sides)
    obj = -np.sum(sides)
    
    # Penalty terms
    penalty = 0.0
    
    # Boundary violations
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # x boundaries
        if cx - s/2 < 0:
            penalty += (cx - s/2)**2
        if cx + s/2 > 1:
            penalty += (cx + s/2 - 1)**2
        # y boundaries
        if cy - s/2 < 0:
            penalty += (cy - s/2)**2
        if cy + s/2 > 1:
            penalty += (cy + s/2 - 1)**2
        # side must be positive
        if s < 0:
            penalty += s**2
    
    # Overlap penalties (smooth approximation)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j])/2
            # If overlapping, penalize the overlap amount
            # Overlap in x: max(0, t - |dx|)
            # Overlap in y: max(0, t - |dy|)
            # We need BOTH to be > 0 for overlap
            # Smooth: use softplus or just square the min of the two gaps
            ox = max(0, t - abs(dx))
            oy = max(0, t - abs(dy))
            # If both positive, there's overlap
            # Penalty: (ox * oy)^2 encourages at least one to be zero
            penalty += (ox * oy)**2
    
    return obj + penalty_weight * penalty

def _solve_ordering_lp(ordering, tol=1e-7, max_time=2.0):
    """
    Solve LP for a complete separation ordering.
    ordering: dict mapping (i,j) -> 0: x_i > x_j, 1: x_j > x_i, 2: y_i > y_j, 3: y_j > y_i
    Returns (centers, sides, value) or None if infeasible.
    """
    n = 10
    c = np.zeros(3*n)
    for i in range(n):
        c[3*i+2] = -1.0
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        row = np.zeros(3*n)
        row[3*i] = -1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        row = np.zeros(3*n)
        row[3*i] = 1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        row = np.zeros(3*n)
        row[3*i+1] = -1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        row = np.zeros(3*n)
        row[3*i+1] = 1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
    
    for i in range(n):
        for j in range(i+1, n):
            sep = ordering.get((i,j))
            if sep is None:
                continue
            if sep == 0:
                row = np.zeros(3*n)
                row[3*i] = 1.0
                row[3*j] = -1.0
                row[3*i+2] = -0.5
                row[3*j+2] = -0.5
                A_ub.append(-row)
                b_ub.append(0.0)
            elif sep == 1:
                row = np.zeros(3*n)
                row[3*j] = 1.0
                row[3*i] = -1.0
                row[3*i+2] = -0.5
                row[3*j+2] = -0.5
                A_ub.append(-row)
                b_ub.append(0.0)
            elif sep == 2:
                row = np.zeros(3*n)
                row[3*i+1] = 1.0
                row[3*j+1] = -1.0
                row[3*i+2] = -0.5
                row[3*j+2] = -0.5
                A_ub.append(-row)
                b_ub.append(0.0)
            elif sep == 3:
                row = np.zeros(3*n)
                row[3*j+1] = 1.0
                row[3*i+1] = -1.0
                row[3*i+2] = -0.5
                row[3*j+2] = -0.5
                A_ub.append(-row)
                b_ub.append(0.0)
    
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = np.empty((0, 3*n))
        b_ub = np.empty(0)
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-9, 1))
    
    start_t = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if time.time() - start_t > max_time:
        return None
    
    if res.status != 0:
        return None
    
    x = res.x
    centers = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
    sides = np.array([x[3*i+2] for i in range(n)])
    
    if not _is_feasible(centers, sides):
        return None
    
    return centers, sides, np.sum(sides)

def _max_side_given_others(centers, sides, i, tol=1e-7):
    """Calculate max side for square i given others are fixed."""
    n = len(sides)
    cx, cy = centers[i]
    s = sides[i]
    
    max_s_boundary = min(2*cx, 2*(1-cx), 2*cy, 2*(1-cy))
    
    max_s = max_s_boundary
    for j in range(n):
        if j == i: continue
        cx_j, cy_j = centers[j]
        s_j = sides[j]
        
        dx = abs(cx - cx_j)
        dy = abs(cy - cy_j)
        
        limit = max(2*dx - s_j, 2*dy - s_j)
        if limit < max_s:
            max_s = limit
            
    return max(0.0, max_s)

def _local_refine(centers, sides, tol=1e-7, max_iter=1000, deadline=None):
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    if deadline is None:
        deadline = time.time() + 100
        
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        improved = False
        
        # 1. Grow each square to max possible size
        for i in range(n):
            new_s = _max_side_given_others(best_centers, best_sides, i, tol)
            if new_s > best_sides[i] + 1e-9:
                best_sides[i] = new_s
                improved = True
        
        # 2. Try to move squares to improve packing
        for i in range(n):
            s = best_sides[i]
            cx, cy = best_centers[i]
            
            for dx in [-0.05, 0.05]:
                for dy in [-0.05, 0.05]:
                    new_cx = cx + dx
                    new_cy = cy + dy
                    if new_cx - s/2 < 0 or new_cx + s/2 > 1: continue
                    if new_cy - s/2 < 0 or new_cy + s/2 > 1: continue
                    
                    new_centers = best_centers.copy()
                    new_centers[i] = [new_cx, new_cy]
                    
                    sim_sides = best_sides.copy()
                    sim_centers = new_centers.copy()
                    
                    for k in range(n):
                        sim_sides[k] = _max_side_given_others(sim_centers, sim_sides, k, tol)
                        
                    if np.sum(sim_sides) > best_val + 1e-7:
                        best_centers = sim_centers
                        best_sides = sim_sides
                        best_val = np.sum(best_sides)
                        improved = True
                        break
                if improved: break
            if improved: break
            
        if not improved:
            break
            
    return best_centers, best_sides

def _branch_and_bound_orderings(centers_init, sides_init, deadline, max_nodes=2000):
    """
    Branch and bound over separation orderings.
    Returns list of complete orderings to try.
    """
    n = 10
    pairs = [(i,j) for i in range(n) for j in range(i+1, n)]
    
    # Sort pairs by how "close" they are in the initial packing
    # Pairs that are close need to be separated, so prioritize them
    def pair_distance(p):
        i, j = p
        dx = abs(centers_init[i, 0] - centers_init[j, 0])
        dy = abs(centers_init[i, 1] - centers_init[j, 1])
        t = (sides_init[i] + sides_init[j])/2
        # If already separated, low priority; if overlapping, high priority
        gap = min(dx, dy) - t
        return gap  # More negative = more overlapping = higher priority
    
    pairs_sorted = sorted(pairs, key=pair_distance)
    
    orderings = []
    
    # Try several random complete orderings, biased toward sensible choices
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        
        ordering = {}
        x_edges = []
        y_edges = []
        
        for (i,j) in pairs_sorted:
            # Choose separation direction
            # Prefer the direction that requires less movement
            dx = centers_init[i, 0] - centers_init[j, 0]
            dy = centers_init[i, 1] - centers_init[j, 1]
            
            # 4 choices: 0: x_i > x_j, 1: x_j > x_i, 2: y_i > y_j, 3: y_j > y_i
            choices = [0, 1, 2, 3]
            
            # Weight choices by how well they align with current positions
            weights = []
            for c in choices:
                if c == 0:  # x_i > x_j
                    w = max(0, dx) + 0.1
                elif c == 1:  # x_j > x_i
                    w = max(0, -dx) + 0.1
                elif c == 2:  # y_i > y_j
                    w = max(0, dy) + 0.1
                else:  # y_j > y_i
                    w = max(0, -dy) + 0.1
                weights.append(w)
            
            # Sample with probability proportional to weight
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            chosen = 0
            for idx, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen = choices[idx]
                    break
            
            ordering[(i,j)] = chosen
            
            # Track edges for cycle detection
            if chosen == 0:
                x_edges.append((i,j))
            elif chosen == 1:
                x_edges.append((j,i))
            elif chosen == 2:
                y_edges.append((i,j))
            else:
                y_edges.append((j,i))
        
        # Check for cycles
        def has_cycle(edges, n):
            adj = [[] for _ in range(n)]
            for (u,v) in edges:
                adj[u].append(v)
            
            WHITE, GRAY, BLACK = 0, 1, 2
            color = [WHITE] * n
            
            def dfs(u):
                color[u] = GRAY
                for v in adj[u]:
                    if color[v] == GRAY:
                        return True
                    if color[v] == WHITE and dfs(v):
                        return True
                color[u] = BLACK
                return False
            
            for u in range(n):
                if color[u] == WHITE:
                    if dfs(u):
                        return True
            return False
        
        if has_cycle(x_edges, n) or has_cycle(y_edges, n):
            continue
        
        orderings.append(ordering)
    
    return orderings

def _gap_filling_pack(random_seed=0):
    """
    Construct a packing using a greedy gap-filling heuristic.
    Returns centers and sides.
    """
    random.seed(random_seed)
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    empty_rects = [(0.0, 0.0, 1.0, 1.0)]
    
    for k in range(n):
        best_rect_idx = -1
        best_size = 0
        best_cx = 0
        best_cy = 0
        
        for idx, (x1, y1, x2, y2) in enumerate(empty_rects):
            w = x2 - x1
            h = y2 - y1
            s = min(w, h)
            if s > best_size:
                best_size = s
                best_rect_idx = idx
                best_cx = (x1 + x2) / 2
                best_cy = (y1 + y2) / 2
            elif s == best_size and s > 0:
                if random.random() < 0.5:
                    best_size = s
                    best_rect_idx = idx
                    best_cx = (x1 + x2) / 2
                    best_cy = (y1 + y2) / 2
        
        if best_rect_idx == -1:
            break
            
        centers[k] = [best_cx, best_cy]
        sides[k] = best_size
        
        x1, y1, x2, y2 = empty_rects[best_rect_idx]
        cx, cy = centers[k]
        s = sides[k]
        
        new_rects = []
        if cx - s/2 > x1:
            new_rects.append((x1, y1, cx - s/2, y2))
        if cx + s/2 < x2:
            new_rects.append((cx + s/2, y1, x2, y2))
        if cy - s/2 > y1:
            new_rects.append((cx - s/2, y1, cx + s/2, cy - s/2))
        if cy + s/2 < y2:
            new_rects.append((cx - s/2, cy + s/2, cx + s/2, y2))
        
        empty_rects[best_rect_idx] = None
        empty_rects = [r for r in empty_rects if r is not None]
        empty_rects.extend(new_rects)
        
    return centers, sides

def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 575
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # 1. Try the known good 3.0 packing
    c0 = np.array([
        [0.25, 0.75],
        [0.75, 0.75],
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
    ])
    s0 = np.array([0.5, 0.5] + [0.25]*8)
    if _is_feasible(c0, s0):
        best_centers = c0.copy()
        best_sides = s0.copy()
        best_val = np.sum(s0)
    
    # 2. Try multiple gap-filling packings with different seeds
    for i in range(15):
        if time.time() > deadline - 120:
            break
            
        c, s = _gap_filling_pack(random_seed=i)
        
        # Discrete refinement
        refine_deadline = min(deadline - 60, time.time() + 8)
        try:
            rc, rs = _local_refine(c, s, max_iter=200, deadline=refine_deadline)
            if _is_feasible(rc, rs):
                val = np.sum(rs)
                if val > best_val:
                    best_val = val
                    best_centers = rc.copy()
                    best_sides = rs.copy()
        except Exception:
            pass
        
        # SLSQP refinement on the best from this seed
        if time.time() < deadline - 40:
            try:
                sc, ss = _slsqp_refine(rc, rs, deadline=deadline - 30)
                if _is_feasible(sc, ss):
                    val = np.sum(ss)
                    if val > best_val:
                        best_val = val
                        best_centers = sc.copy()
                        best_sides = ss.copy()
            except Exception:
                pass
    
    # 3. Branch-and-bound ordering LP on the best packing
    if best_centers is not None and time.time() < deadline - 60:
        try:
            orderings = _branch_and_bound_orderings(best_centers, best_sides, deadline)
            for ordering in orderings:
                if time.time() > deadline - 30:
                    break
                result = _solve_ordering_lp(ordering, max_time=3.0)
                if result is not None:
                    centers, sides, val = result
                    if val > best_val:
                        best_val = val
                        best_centers = centers.copy()
                        best_sides = sides.copy()
        except Exception:
            pass
    
    # 4. Final SLSQP polish on the best found
    if best_centers is not None and time.time() < deadline - 10:
        try:
            sc, ss = _slsqp_refine(best_centers, best_sides, deadline=deadline - 5)
            if _is_feasible(sc, ss):
                val = np.sum(ss)
                if val > best_val:
                    best_val = val
                    best_centers = sc.copy()
                    best_sides = ss.copy()
        except Exception:
            pass
        
        # One more discrete refine
        if time.time() < deadline - 3:
            try:
                rc, rs = _local_refine(best_centers, best_sides, max_iter=200, deadline=deadline - 2)
                if _is_feasible(rc, rs):
                    val = np.sum(rs)
                    if val > best_val:
                        best_val = val
                        best_centers = rc.copy()
                        best_sides = rs.copy()
            except Exception:
                pass
    
    return best_centers, best_sides
