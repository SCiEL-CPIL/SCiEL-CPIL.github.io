# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49, p = 7).
    Strategy: Local Search with smooth surrogate objective.
    We maintain a set E (excluded points). N = F_49^2 \ E.
    Constraint: For every point x in F_49^2, there exists a direction v 
    such that line(x, v) \ {x} is disjoint from E.
    Let S(x) be the set of directions v for which line(x, v) \ {x} is disjoint from E.
    The constraint is satisfied iff for all x, |S(x)| > 0.
    
    We want to maximize |E|.
    
    We define a "slack" for each point x as the minimum number of "conflicts" 
    (points in E on the line) over all directions v.
    Actually, let's define:
    For a point x, let c(x, v) be the number of points in E on line(x, v) excluding x.
    The condition is min_v c(x, v) == 0.
    
    To guide local search, we can look at points x that are "tight" (min_v c(x, v) is small).
    If we add a point z to E, it increases c(x, v) for all (x, v) where z is on line(x, v).
    This might make min_v c(x, v) > 0 for some x, violating the constraint.
    
    We use a greedy local search:
    1. Start with a valid E (e.g., empty, or a greedy construction).
    2. Try to add points to E. If adding z violates the constraint (makes some x have min_v c(x, v) > 0), 
       we might try to "fix" it by removing some other point from E, or just skip z.
    3. To improve over pure greedy, we can use a "lookahead" or "swap" strategy.
    
    However, the prompt suggests a smooth surrogate. 
    Let's define an energy function E_energy = sum_x max(0, min_v c(x, v) - 0). 
    This is 0 if valid. We want to minimize this energy while maximizing |E|.
    Since validity is a hard constraint (binary), we can't really use a soft surrogate for the constraint 
    unless we allow "soft" violations and penalize them, but the evaluator requires strict validity.
    
    So, we stick to strict validity but use a smarter search order or local moves.
    The current program uses a greedy "try to add" approach with random restarts.
    
    Improvement idea: 
    Instead of just "try to add", we can perform "2-opt" swaps: 
    Remove a point z1 from E and add z2 to E if it results in a valid state.
    Or, simply a more efficient greedy with backtracking?
    
    Let's try a different angle: 
    The structure of F_49^2 allows for algebraic constructions.
    But we are asked to stick to the design family (search-based).
    
    Let's optimize the current greedy by:
    1. Precomputing the "impact" of adding a point z.
    2. Using a priority queue to select the "best" point to add next.
       "Best" could be defined as the point whose addition causes the least "damage" 
       to the slack of other points.
    
    Slack of point x: s(x) = min_v c(x, v).
    Initially s(x) = 0 for all x.
    When we add z, for each (x, v) affected, c(x, v) increases.
    This might increase s(x) if the minimum was attained by that v.
    
    We want to keep s(x) = 0 for all x.
    A point z is "safe" to add if for all affected x, there exists another v' such that c(x, v') = 0.
    
    Let's implement a fast check for "safe to add".
    
    Precomputation:
    - line_other[x, v]: list of points on line(x, v) excluding x.
    - affected_by[z]: list of (x, v) such that z is in line_other[x, v].
    
    State:
    - in_E: boolean array of size 2401.
    - c: 2D array (2401, 50) of counts.
    - min_c: 1D array (2401) of min(c[x, :]).
    - count_min: 1D array (2401) of how many v achieve min_c[x].
    
    Update when adding z:
    For each (x, v) in affected_by[z]:
        old_c = c[x, v]
        c[x, v] += 1
        if old_c == min_c[x]:
            # The minimum might increase
            # Recalculate min_c[x] and count_min[x]
            # This is expensive if done naively.
            # Optimization: Since c[x, v] only increases, min_c[x] can only increase or stay same.
            # We can maintain a heap or just scan if 50 is small. 50 is small.
            pass
    
    Given q=49, num_dirs=50, total_points=2401.
    Scanning 50 elements is very fast.
    
    Let's implement this "Safe Add" greedy with multiple restarts and random orders.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    num_dirs = q + 1  # 50

    deadline = time.time() + 570.0

    # Field arithmetic setup (same as before)
    k_const = 3
    elem_to_idx = {}
    idx_to_elem = []
    for a in range(p):
        for b in range(p):
            idx = a * p + b
            elem_to_idx[(a, b)] = idx
            idx_to_elem.append((a, b))

    def f_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def f_mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        aa = (a1 * a2 + b1 * b2 * k_const) % p
        bb = (a1 * b2 + b1 * a2) % p
        return (aa, bb)

    def point_to_indices(idx):
        return idx // q, idx % q

    def indices_to_point(ix, iy):
        return ix * q + iy

    # Directions
    directions = []
    directions.append(((0, 0), (1, 0)))
    for m_idx in range(q):
        m_elem = idx_to_elem[m_idx]
        directions.append(((1, 0), m_elem))

    # Precompute line_other
    line_other = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    for pidx in range(total_points):
        xi, yi = point_to_indices(pidx)
        x_elem = idx_to_elem[xi]
        y_elem = idx_to_elem[yi]
        for didx, (dx, dy) in enumerate(directions):
            count = 0
            for t_idx in range(1, q): # t != 0
                t_elem = idx_to_elem[t_idx]
                tdx = f_mul(t_elem, dx)
                tdy = f_mul(t_elem, dy)
                nx = f_add(x_elem, tdx)
                ny = f_add(y_elem, tdy)
                nxi = elem_to_idx[nx]
                nyi = elem_to_idx[ny]
                npidx = indices_to_point(nxi, nyi)
                line_other[pidx, didx, count] = npidx
                count += 1

    # Precompute affected_by
    # affected_by[z] = list of (x, v) such that z is in line_other[x, v]
    affected_by = [[] for _ in range(total_points)]
    for x in range(total_points):
        for v in range(num_dirs):
            for z in line_other[x, v]:
                affected_by[z].append((x, v))

    # State management for local search
    in_E = np.zeros(total_points, dtype=bool)
    c = np.zeros((total_points, num_dirs), dtype=np.int32)
    min_c = np.zeros(total_points, dtype=np.int32)
    count_min = np.ones(total_points, dtype=np.int32) # Number of v with c[x,v] == min_c[x]

    def update_min_c(x):
        """Recalculate min_c[x] and count_min[x] from c[x, :]"""
        row = c[x]
        m = np.min(row)
        cnt = np.sum(row == m)
        min_c[x] = m
        count_min[x] = cnt

    def try_add(z):
        """
        Try to add point z to E.
        Returns True if successful, False otherwise.
        Updates state if successful.
        """
        if in_E[z]:
            return False

        # Check if adding z keeps all points valid.
        # A point x is valid if min_c[x] == 0.
        # Adding z increases c[x, v] for (x, v) in affected_by[z].
        # We need to check if any x becomes invalid (min_c[x] becomes > 0).
        
        # Optimization: Only points x where z affects the minimum are at risk.
        # If for a point x, the direction v affected by z has c[x, v] > min_c[x], 
        # then min_c[x] doesn't change.
        # If c[x, v] == min_c[x], then min_c[x] might increase.
        
        # Let's collect the set of x that might be affected.
        affected_x = set()
        for (x, v) in affected_by[z]:
            if c[x, v] == min_c[x]:
                affected_x.add(x)
        
        # If no x is affected in this way, adding z is safe.
        # If some x is affected, we need to check if min_c[x] becomes > 0.
        
        # Tentatively apply changes
        changes = []
        for (x, v) in affected_by[z]:
            changes.append((x, v, c[x, v]))
            c[x, v] += 1
        
        # Check validity for affected x
        valid = True
        for x in affected_x:
            update_min_c(x)
            if min_c[x] > 0:
                valid = False
                break
        
        if valid:
            in_E[z] = True
            return True
        else:
            # Revert
            for (x, v, old_val) in changes:
                c[x, v] = old_val
                if x in affected_x:
                    update_min_c(x)
            return False

    def try_remove(z):
        """
        Remove point z from E. Always valid to do so if z is in E.
        """
        if not in_E[z]:
            return False
        in_E[z] = False
        for (x, v) in affected_by[z]:
            c[x, v] -= 1
            if c[x, v] < min_c[x]:
                update_min_c(x)
        return True

    # Search Loop
    best_E = np.zeros(total_points, dtype=bool)
    best_size = 0

    num_restarts = 100 # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Reset state
        in_E[:] = False
        c[:] = 0
        min_c[:] = 0
        count_min[:] = 1
        
        size = 0
        
        # Random order
        order = list(range(total_points))
        random.Random(1000 + restart).shuffle(order)
        
        # Greedy add
        for z in order:
            if time.time() > deadline:
                break
            if try_add(z):
                size += 1
        
        if size > best_size:
            best_size = size
            best_E[:] = in_E
            
        # Local Search: Try swaps
        # Try to remove a point and add a different one
        # This is complex to implement efficiently within the time limit without heavy overhead.
        # Given the time limit, simple greedy with many restarts is often competitive.
        # Let's add a simple "second pass" with a different order if time permits.
        
        if time.time() < deadline - 30:
            # Try adding points in a different order (e.g., reverse)
            # But state is already full of added points.
            # We can try to "shake" the solution: remove a random point, try to add others.
            # This is risky and might not improve.
            pass

    # Build result
    points_list = []
    for pidx in range(total_points):
        if not best_E[pidx]:
            xi, yi = point_to_indices(pidx)
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        points_list = []
        for pidx in range(total_points):
            xi, yi = point_to_indices(pidx)
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)

# EVOLVE-BLOCK-END
