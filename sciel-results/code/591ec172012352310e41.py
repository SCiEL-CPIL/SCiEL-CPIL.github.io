"""
Minimum maximal no-isosceles set in an N x N grid.

Strategy:
1. Precompute conflict masks for fast O(k) add-checks using bitmasks.
2. Randomized greedy construction with varied point orderings.
3. Aggressive minimality reduction (remove points while maintaining maximality).
4. Coverage-based greedy to construct sets that block many points.
5. Local search: try swapping one point for another to reduce set size.
6. MILP-based refinement: formulate as a set cover problem to find minimal
   subset that maintains maximality.
"""
import time
import random
import itertools
import numpy as np

N = 16
DEADLINE = 320.0

# Precompute all points
ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
N_POINTS = len(ALL_POINTS)

# Precompute distance squared matrix
def _precompute_dist2():
    d2 = np.zeros((N_POINTS, N_POINTS), dtype=np.int32)
    for i in range(N_POINTS):
        x1, y1 = ALL_POINTS[i]
        for j in range(i + 1, N_POINTS):
            x2, y2 = ALL_POINTS[j]
            d = (x1 - x2)**2 + (y1 - y2)**2
            d2[i, j] = d
            d2[j, i] = d
    return d2

DIST2 = _precompute_dist2()

def _dist2_idx(i, j):
    return DIST2[i, j]

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _point_to_idx(p):
    return p[1] * N + p[0]

def _isosceles_idx(i, j, k):
    d1 = DIST2[i, j]
    d2 = DIST2[i, k]
    d3 = DIST2[j, k]
    return d1 == d2 or d1 == d3 or d2 == d3

# Precompute conflict masks:
# conflict_mask[p][a] = bitmask of points b such that (p, a, b) forms an isosceles triangle
def _precompute_conflict_masks():
    cm = [[0] * N_POINTS for _ in range(N_POINTS)]
    for p in range(N_POINTS):
        for a in range(N_POINTS):
            if p == a:
                continue
            mask = 0
            d_pa = DIST2[p, a]
            for b in range(N_POINTS):
                if b == p or b == a:
                    continue
                d_pb = DIST2[p, b]
                d_ab = DIST2[a, b]
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    mask |= (1 << b)
            cm[p][a] = mask
    return cm

CONFLICT_MASK = _precompute_conflict_masks()

def _can_add_fast(p_idx, chosen_mask, chosen_list):
    """
    Check if point p_idx can be added to chosen without creating isosceles triangle.
    Uses precomputed conflict masks for O(k) check instead of O(k^2).
    chosen_mask: bitmask of chosen points
    chosen_list: list of chosen point indices
    """
    for a in chosen_list:
        if (chosen_mask & CONFLICT_MASK[p_idx][a]) != 0:
            return False
    return True

def _is_maximal_fast(chosen_mask, chosen_list):
    """Check if chosen set is maximal using fast conflict masks."""
    for q in range(N_POINTS):
        if (chosen_mask >> q) & 1:
            continue
        if _can_add_fast(q, chosen_mask, chosen_list):
            return False
    return True

def _reduce_to_minimal(chosen_set, all_indices):
    """
    Iteratively remove points from chosen_set if the remaining set is still maximal.
    """
    chosen_list = list(chosen_set)
    changed = True
    while changed:
        changed = False
        for i in range(len(chosen_list)):
            p = chosen_list[i]
            S_test = chosen_list[:i] + chosen_list[i+1:]
            
            if len(S_test) < 2:
                continue
            
            S_test_mask = 0
            for x in S_test:
                S_test_mask |= (1 << x)
            
            # Check maximality using fast method
            maximal = True
            for q in all_indices:
                if (S_test_mask >> q) & 1:
                    continue
                if _can_add_fast(q, S_test_mask, S_test):
                    maximal = False
                    break
            
            if maximal:
                chosen_list = S_test
                changed = True
                break
                
    return chosen_list

def _greedy_random(order, rng):
    """
    Greedy construction: iterate through points in given order, add if valid.
    """
    chosen = []
    chosen_mask = 0
    for p in order:
        if (chosen_mask >> p) & 1:
            continue
        if _can_add_fast(p, chosen_mask, chosen):
            chosen.append(p)
            chosen_mask |= (1 << p)
    return chosen

def _coverage_score(p_idx, chosen, all_indices, chosen_mask):
    """
    Score how many points p_idx would block if added to chosen.
    """
    if not _can_add_fast(p_idx, chosen_mask, chosen):
        return 0
    
    new_blocked = 0
    for q in all_indices:
        if (chosen_mask >> q) & 1 or q == p_idx:
            continue
        
        # Check if q is already blocked by chosen
        already_blocked = False
        for a in chosen:
            if (chosen_mask & CONFLICT_MASK[q][a]) != 0:
                already_blocked = True
                break
        
        if already_blocked:
            continue
        
        # Check if adding p_idx would block q
        # q is blocked by (p_idx, q, a) if isosceles, i.e., a in CONFLICT_MASK[q][p_idx]
        if (1 << p_idx) & CONFLICT_MASK[q][0] if False else False:
            pass
        # Better: check if p_idx is in conflict with q for some a in chosen
        # Actually, (q, p_idx, a) is isosceles iff a is in CONFLICT_MASK[q][p_idx]
        # But we need a in chosen such that (q, p_idx, a) is isosceles
        # That means a in CONFLICT_MASK[q][p_idx] AND a in chosen
        if (chosen_mask & CONFLICT_MASK[q][p_idx]) != 0:
            new_blocked += 1
            
    return new_blocked

def _greedy_coverage(all_indices, rng):
    """
    Greedy construction using coverage heuristic.
    """
    remaining = list(all_indices)
    chosen = []
    chosen_mask = 0
    
    if not remaining:
        return chosen
    
    start = rng.choice(remaining)
    remaining.remove(start)
    chosen.append(start)
    chosen_mask |= (1 << start)
    
    while remaining:
        best_p = None
        best_score = -1
        
        candidates = remaining
        if len(candidates) > 150:
            candidates = rng.sample(candidates, 150)
        
        for p in candidates:
            score = _coverage_score(p, chosen, all_indices, chosen_mask)
            if score > best_score:
                best_score = score
                best_p = p
        
        if best_p is None or best_score <= 0:
            break
            
        remaining.remove(best_p)
        chosen.append(best_p)
        chosen_mask |= (1 << best_p)
        
    return chosen

def _local_search(chosen, all_indices, time_limit=30.0):
    """
    Try to improve chosen by swapping one point for another.
    """
    start_time = time.time()
    chosen_list = list(chosen)
    n = len(chosen_list)
    
    for i in range(n):
        if time.time() - start_time > time_limit:
            break
        
        p_remove = chosen_list[i]
        S_without = chosen_list[:i] + chosen_list[i+1:]
        
        S_without_mask = 0
        for x in S_without:
            S_without_mask |= (1 << x)
        
        for p_add in all_indices:
            if (S_without_mask >> p_add) & 1:
                continue
            if p_add == p_remove:
                continue
            
            if not _can_add_fast(p_add, S_without_mask, S_without):
                continue
            
            S_new = S_without + [p_add]
            S_new_mask = S_without_mask | (1 << p_add)
            
            # Check if maximal
            maximal = True
            for q in all_indices:
                if (S_new_mask >> q) & 1:
                    continue
                if _can_add_fast(q, S_new_mask, S_new):
                    maximal = False
                    break
            
            if maximal:
                reduced = _reduce_to_minimal(S_new, all_indices)
                if len(reduced) < n:
                    return reduced
                    
    return None

def _milp_minimize(chosen, all_indices, time_limit=30.0):
    """
    Use MILP to find minimal subset of chosen that is still maximal.
    For each point q not in chosen, we need at least one triple (q, a, b) with a,b in chosen
    that is isosceles. This is a set cover problem.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    start_time = time.time()
    chosen_set = set(chosen)
    n_chosen = len(chosen)
    
    if n_chosen < 3:
        return chosen[:]
    
    # For each point q not in chosen, find all pairs (a, b) in chosen such that (q, a, b) is isosceles
    # We need to select a subset of chosen such that for every q, at least one such pair exists.
    
    # Build the problem:
    # Variables: x_i for i in range(n_chosen) (1 if chosen[i] is kept)
    # For each q not in chosen: need sum over pairs (i,j) where (q, chosen[i], chosen[j]) is isosceles of y_{q,i,j} >= 1
    # where y_{q,i,j} = x_i * x_j (linearized)
    
    # This can be large, so let's use a simpler heuristic first and MILP if feasible
    best = chosen[:]
    best_set = set(best)
    
    # First try iterative removal
    for round in range(15):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(len(best)):
            p = best[i]
            candidate = best[:i] + best[i+1:]
            if len(candidate) < 2:
                continue
            cand_mask = 0
            for x in candidate:
                cand_mask |= (1 << x)
            maximal = True
            for q in all_indices:
                if (cand_mask >> q) & 1:
                    continue
                if _can_add_fast(q, cand_mask, candidate):
                    maximal = False
                    break
            if maximal:
                best = candidate
                best_set = set(best)
                improved = True
                break
        if not improved:
            break
            
    # Now try MILP on the reduced set if it's small enough
    if time.time() - start_time < time_limit - 5 and len(best) <= 12:
        try:
            n_vars = len(best)
            # For each q not in best, find pairs (i,j) in best that block q
            blocking_pairs = {}  # q -> list of (i, j)
            for q in all_indices:
                if q in best_set:
                    continue
                pairs = []
                for i in range(n_vars):
                    for j in range(i + 1, n_vars):
                        a = best[i]
                        b = best[j]
                        if _isosceles_idx(q, a, b):
                            pairs.append((i, j))
                if pairs:
                    blocking_pairs[q] = pairs
            
            if not blocking_pairs:
                return best
            
            # MILP formulation:
            # x_i in {0,1} for i in range(n_vars)
            # y_{q,i,j} in {0,1} for each q in blocking_pairs, (i,j) in blocking_pairs[q]
            # Constraints:
            #   y_{q,i,j} <= x_i
            #   y_{q,i,j} <= x_j  
            #   y_{q,i,j} >= x_i + x_j - 1
            #   sum_{(i,j) in blocking_pairs[q]} y_{q,i,j} >= 1 for each q
            # Objective: minimize sum(x_i)
            
            n_q = len(blocking_pairs)
            n_y = sum(len(v) for v in blocking_pairs.values())
            n_total = n_vars + n_y
            
            # Objective: minimize sum of x_i
            c = np.zeros(n_total)
            c[:n_vars] = 1.0
            
            # Constraints
            A_rows = []
            lb = []
            ub = []
            
            for q_idx, (q, pairs) in enumerate(blocking_pairs.items()):
                # sum of y_{q,i,j} >= 1
                row = np.zeros(n_total)
                y_offset = sum(len(blocking_pairs[qq]) for qq in list(blocking_pairs.keys())[:q_idx])
                for k, (i, j) in enumerate(pairs):
                    row[n_vars + y_offset + k] = 1.0
                A_rows.append(row)
                lb.append(1.0)
                ub.append(np.inf)
                
                # For each (i,j) in pairs: y <= x_i, y <= x_j, y >= x_i + x_j - 1
                for k, (i, j) in enumerate(pairs):
                    y_idx = n_vars + y_offset + k
                    # y - x_i <= 0
                    row1 = np.zeros(n_total)
                    row1[y_idx] = 1.0
                    row1[i] = -1.0
                    A_rows.append(row1)
                    lb.append(-np.inf)
                    ub.append(0.0)
                    
                    # y - x_j <= 0
                    row2 = np.zeros(n_total)
                    row2[y_idx] = 1.0
                    row2[j] = -1.0
                    A_rows.append(row2)
                    lb.append(-np.inf)
                    ub.append(0.0)
                    
                    # -y + x_i + x_j >= -1  =>  y - x_i - x_j <= 1
                    row3 = np.zeros(n_total)
                    row3[y_idx] = 1.0
                    row3[i] = -1.0
                    row3[j] = -1.0
                    A_rows.append(row3)
                    lb.append(-np.inf)
                    ub.append(1.0)
            
            if A_rows:
                A = np.array(A_rows)
                constraints = LinearConstraint(A, lb, ub)
                integrality = np.zeros(n_total)
                integrality[:] = 1  # all binary
                bounds = Bounds(0, 1)
                
                res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                           options={'time_limit': max(1, time_limit - (time.time() - start_time) - 2)})
                
                if res.status == 0 or res.status == 1:
                    x_vals = res.x[:n_vars]
                    new_best = [best[i] for i in range(n_vars) if x_vals[i] > 0.5]
                    if len(new_best) < len(best):
                        # Verify
                        new_best_set = set(new_best)
                        new_best_mask = 0
                        for x in new_best:
                            new_best_mask |= (1 << x)
                        maximal = True
                        for q in all_indices:
                            if (new_best_mask >> q) & 1:
                                continue
                            if _can_add_fast(q, new_best_mask, new_best):
                                maximal = False
                                break
                        if maximal:
                            best = new_best
                            best_set = new_best_set
        except Exception:
            pass
            
    return best

def construct():
    start_time = time.time()
    
    all_indices = list(range(N_POINTS))
    
    best_set = None
    best_size = N_POINTS + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Random greedy + reduction
    for attempt in range(500):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        order = all_indices[:]
        rng.shuffle(order)
        
        chosen = _greedy_random(order, rng)
        
        # Ensure maximality by adding more points if needed
        chosen_mask = 0
        for x in chosen:
            chosen_mask |= (1 << x)
        changed = True
        while changed:
            changed = False
            for p in all_indices:
                if (chosen_mask >> p) & 1:
                    continue
                if _can_add_fast(p, chosen_mask, chosen):
                    chosen.append(p)
                    chosen_mask |= (1 << p)
                    changed = True
        
        reduced = _reduce_to_minimal(chosen, all_indices)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 7:
            break
    
    # Strategy 2: Coverage greedy + reduction
    for attempt in range(200):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        chosen = _greedy_coverage(all_indices, rng)
        
        if not chosen:
            continue
            
        chosen_mask = 0
        for x in chosen:
            chosen_mask |= (1 << x)
        # Ensure maximality
        changed = True
        while changed:
            changed = False
            for p in all_indices:
                if (chosen_mask >> p) & 1:
                    continue
                if _can_add_fast(p, chosen_mask, chosen):
                    chosen.append(p)
                    chosen_mask |= (1 << p)
                    changed = True
        
        reduced = _reduce_to_minimal(chosen, all_indices)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 7:
            break
    
    # Strategy 3: Local search on best found so far
    if best_set is not None:
        for ls_round in range(5):
            if time.time() - start_time > DEADLINE - 15:
                break
            improved = _local_search(best_set, all_indices, time_limit=20.0)
            if improved is not None and len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            if best_size <= 7:
                break
    
    # Strategy 4: MILP-based minimization
    if best_set is not None:
        if time.time() - start_time < DEADLINE - 10:
            minimized = _milp_minimize(best_set, all_indices, time_limit=15.0)
            if len(minimized) < best_size:
                best_size = len(minimized)
                best_set = minimized[:]
    
    if best_set is None:
        # Fallback: simple greedy
        chosen = _greedy_random(all_indices[:], rng)
        return [ALL_POINTS[i] for i in chosen]
    
    return [ALL_POINTS[i] for i in best_set]
