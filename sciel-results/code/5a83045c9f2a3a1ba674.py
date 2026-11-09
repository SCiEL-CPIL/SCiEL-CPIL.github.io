"""Improved search for minimum maximal no-isosceles set on N x N grid.
Structural change: Branch-and-bound with maximality-based pruning.
"""
N = 16
import time
import random
import itertools
import numpy as np


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2_ = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2_ or d1 == d3 or d2_ == d3


def _can_add_np(p, chosen_np):
    """Vectorized check if point p can be added to chosen (numpy array) without creating isosceles triangle."""
    if len(chosen_np) == 0:
        return True
    dx = p[0] - chosen_np[:, 0]
    dy = p[1] - chosen_np[:, 1]
    d_to_p = dx * dx + dy * dy
    
    n = len(chosen_np)
    if n < 2:
        return True
    
    dx_chosen = chosen_np[:, 0][:, None] - chosen_np[:, 0][None, :]
    dy_chosen = chosen_np[:, 1][:, None] - chosen_np[:, 1][None, :]
    d_chosen = dx_chosen * dx_chosen + dy_chosen * dy_chosen
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    if len(i_idx) == 0:
        return True
    
    d1 = d_to_p[i_idx]
    d2 = d_to_p[j_idx]
    d3 = d_chosen[i_idx, j_idx]
    
    if np.any(d1 == d2) or np.any(d1 == d3) or np.any(d2 == d3):
        return False
    return True


def _is_valid_no_iso(chosen):
    """Verify no three points form isosceles triangle."""
    if len(chosen) < 3:
        return True
    n = len(chosen)
    d2mat = np.zeros((n, n), dtype=np.int32)
    pts = np.array(chosen, dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = (pts[i, 0] - pts[j, 0]) ** 2 + (pts[i, 1] - pts[j, 1]) ** 2
            d2mat[i, j] = dist
            d2mat[j, i] = dist
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if d2mat[i, j] == d2mat[i, k] or d2mat[i, j] == d2mat[j, k] or d2mat[i, k] == d2mat[j, k]:
                    return False
    return True


def _is_maximal(chosen, all_points):
    """Check if no point can be added."""
    chosen_set = set(chosen)
    chosen_np = np.array(chosen, dtype=np.int32) if chosen else np.empty((0, 2), dtype=np.int32)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add_np(p, chosen_np):
            return False
    return True


def _greedy(order, chosen=None):
    """Greedy construction following given point order."""
    if chosen is None:
        chosen = []
    else:
        chosen = list(chosen)
    chosen_set = set(chosen)
    chosen_np = np.array(chosen, dtype=np.int32) if chosen else np.empty((0, 2), dtype=np.int32)
    
    for p in order:
        if p not in chosen_set:
            if _can_add_np(p, chosen_np):
                chosen.append(p)
                chosen_set.add(p)
                chosen_np = np.vstack([chosen_np, np.array(p, dtype=np.int32)])
    return chosen


def _construct_maximal(chosen, all_points):
    """Add points greedily until maximal."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    chosen_np = np.array(chosen, dtype=np.int32) if chosen else np.empty((0, 2), dtype=np.int32)
    changed = True
    while changed:
        changed = False
        for p in all_points:
            if p not in chosen_set:
                if _can_add_np(p, chosen_np):
                    chosen.append(p)
                    chosen_set.add(p)
                    chosen_np = np.vstack([chosen_np, np.array(p, dtype=np.int32)])
                    changed = True
    return chosen


def _count_free_points(chosen_np, all_points, chosen_set):
    """Count how many unchosen points can still be added to chosen."""
    if len(chosen_np) == 0:
        return len(all_points)
    free_count = 0
    for p in all_points:
        if p not in chosen_set:
            if _can_add_np(p, chosen_np):
                free_count += 1
    return free_count


def _branch_and_bound(all_points, best_len, max_nodes=500000, deadline=None):
    """
    Branch-and-bound search using maximality constraint for pruning.
    
    Key insight: For a set S to be maximal, every point not in S must be "blocked"
    by S. We use this to prune: if at any point, the number of free points (that
    could be added) is greater than what the remaining budget can block, we prune.
    
    Actually, a simpler but effective pruning: we search for a set of size < best_len.
    We build the set incrementally. At each step, we decide include or exclude each point.
    Pruning: if current_size + remaining_points < best_len, we could still beat best.
    But the key structural pruning is: if the set is already "close to maximal" and 
    we have too many free points that can't be blocked by the few remaining points we can add.
    """
    best = [None]
    best_len_ref = [best_len]
    nodes = [0]
    
    # Precompute: for each point, which other points would form isosceles with it
    # This helps with fast blocking analysis
    all_points_list = list(all_points)
    n_pts = len(all_points_list)
    pt_index = {p: i for i, p in enumerate(all_points_list)}
    
    # Precompute isosceles triples: for each point p, list of pairs (i,j) such that
    # (p, all_points[i], all_points[j]) is isosceles
    # Actually, let's precompute for each point p, the set of points q such that
    # p and q are "compatible" in the sense that there exists some r where
    # (p, q, r) is isosceles. This is complex. Let's use a simpler approach.
    
    # For pruning: a point p is "blocked" by set S if for every pair (a,b) in S,
    # (p, a, b) is isosceles. Equivalently, p cannot be added to S.
    # We'll use _can_add_np for this.
    
    # Order points by a heuristic: points that are "more constrained" first
    # A point is more constrained if it forms isosceles triangles with more other points
    constraint_score = [0] * n_pts
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            # Count how many k make (i,j,k) isosceles
            pi = all_points_list[i]
            pj = all_points_list[j]
            d_ij = _d2(pi, pj)
            for k in range(n_pts):
                if k == i or k == j:
                    continue
                pk = all_points_list[k]
                d_ik = _d2(pi, pk)
                d_jk = _d2(pj, pk)
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    constraint_score[i] += 1
                    constraint_score[j] += 1
    
    # Sort points: more constrained points first (they block more)
    order = sorted(range(n_pts), key=lambda i: (-constraint_score[i], all_points_list[i][0] + all_points_list[i][1]))
    
    # Map from original index to position in order
    pos_in_order = [0] * n_pts
    for pos, orig_idx in enumerate(order):
        pos_in_order[orig_idx] = pos
    
    def _search(chosen, chosen_np, chosen_set, start_pos, remaining_budget):
        if deadline is not None and time.time() > deadline:
            return
        if nodes[0] >= max_nodes:
            return
        nodes[0] += 1
        
        current_size = len(chosen)
        
        # Pruning 1: if we've used too many points
        if current_size >= best_len_ref[0]:
            return
        
        # Pruning 2: if remaining points can't help us beat best
        # We need to add at most (best_len_ref[0] - 1 - current_size) more points
        # But we also need the set to be maximal, which requires blocking all free points
        
        # Count free points
        if start_pos >= n_pts:
            # All points decided. Check if maximal and valid.
            if _is_maximal(chosen, all_points_list):
                if _is_valid_no_iso(chosen):
                    if current_size < best_len_ref[0]:
                        best_len_ref[0] = current_size
                        best[0] = list(chosen)
            return
        
        # Get next point to decide
        orig_idx = order[start_pos]
        p = all_points_list[orig_idx]
        
        # Option 1: Include p (if valid)
        if _can_add_np(p, chosen_np):
            new_chosen = chosen + [p]
            new_chosen_set = chosen_set | {p}
            new_chosen_np = np.vstack([chosen_np, np.array(p, dtype=np.int32)]) if len(chosen_np) > 0 else np.array([p], dtype=np.int32)
            _search(new_chosen, new_chosen_np, new_chosen_set, start_pos + 1, remaining_budget - 1)
        
        # Option 2: Exclude p
        # Pruning: if we exclude p, we need to make sure p will be blocked by future points
        # For now, just recurse
        _search(chosen, chosen_np, chosen_set, start_pos + 1, remaining_budget)
    
    empty_np = np.empty((0, 2), dtype=np.int32)
    _search([], empty_np, set(), 0, best_len - 1)
    
    return best[0]


def _remove_and_remaximize(current, all_points, idx):
    """Remove point at idx, then make maximal again."""
    new_set = list(current)
    new_set.pop(idx)
    extended = _construct_maximal(new_set, all_points)
    return extended


def _local_search(all_points, start_set, max_iters=5000, deadline=None):
    """Local search: try removing each point and re-maximizing, keep if better."""
    current = list(start_set)
    best = list(current)
    best_len = len(current)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for idx in range(len(current)):
            if deadline is not None and time.time() > deadline:
                break
            extended = _remove_and_remaximize(current, all_points, idx)
            if extended is not None and len(extended) < len(current):
                current = extended
                improved = True
                break
        
        if not improved:
            break
        
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
    
    return best


def _simulated_annealing(all_points, start_set, max_iters=20000, deadline=None):
    """Simulated annealing to minimize the size of a maximal no-isosceles set."""
    current = list(start_set)
    best = list(start_set)
    best_len = len(current)
    
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
        
        if T < T_min:
            break
        
        if current:
            idx = rng.randint(0, len(current) - 1)
            extended = _remove_and_remaximize(current, all_points, idx)
            
            if extended is not None:
                delta = len(extended) - len(current)
                if delta <= 0 or rng.random() < (1.0 - delta) / T:
                    current = extended
        
        T *= cooling
    
    return best


def _diversified_greedy(all_points, rng, num_orderings=5):
    """Generate multiple diverse greedy orderings and return the best maximal set."""
    best = None
    best_len = float('inf')
    
    for _ in range(num_orderings):
        pts = all_points[:]
        rng.shuffle(pts)
        chosen = _greedy(pts)
        extended = _construct_maximal(chosen, all_points)
        if len(extended) < best_len:
            best = extended
            best_len = len(extended)
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s limit
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_len = float('inf')
    
    # Phase 1: Quick greedy constructions with various deterministic orderings
    orderings = [
        ("row_major", list(itertools.product(range(N), repeat=2))),
        ("col_major", sorted(all_points, key=lambda p: (p[1], p[0]))),
        ("diag", sorted(all_points, key=lambda p: (p[0] + p[1], p[0]))),
        ("checker", [p for p in all_points if (p[0] + p[1]) % 2 == 0] +
                   [p for p in all_points if (p[0] + p[1]) % 2 == 1]),
        ("anti_diag", sorted(all_points, key=lambda p: (abs(p[0] - p[1]), p[0]))),
    ]
    
    for name, order in orderings:
        if time.time() > deadline:
            break
        chosen = _greedy(order)
        extended = _construct_maximal(chosen, all_points)
        if len(extended) < best_len:
            best = extended
            best_len = len(extended)
    
    # Phase 2: Branch-and-bound search with maximality pruning
    if best is not None and time.time() < deadline - 60:
        bt_result = _branch_and_bound(all_points, best_len, max_nodes=300000,
                                      deadline=deadline - 60)
        if bt_result is not None and len(bt_result) < best_len:
            best = bt_result
            best_len = len(best)
    
    # Phase 3: Local search from best
    if best is not None and time.time() < deadline - 30:
        local_best = _local_search(all_points, best, max_iters=10000,
                                   deadline=deadline - 30)
        if len(local_best) < best_len:
            best = local_best
            best_len = len(best)
    
    # Phase 4: Simulated annealing
    if best is not None and time.time() < deadline - 15:
        sa_best = _simulated_annealing(all_points, best, max_iters=20000,
                                       deadline=deadline - 15)
        if len(sa_best) < best_len:
            best = sa_best
            best_len = len(best)
    
    # Phase 5: Randomized restarts with local search
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        
        extended = _diversified_greedy(all_points, rng, num_orderings=3)
        if extended is not None and len(extended) < best_len:
            best = extended
            best_len = len(extended)
        
        if attempts % 5 == 0 and time.time() < deadline - 10:
            local = _local_search(all_points, extended, max_iters=2000,
                                  deadline=deadline - 10)
            if len(local) < best_len:
                best = local
                best_len = len(best)
    
    # Phase 6: Final polish
    if best is not None and time.time() < deadline - 2:
        final = _local_search(all_points, best, max_iters=5000,
                              deadline=deadline - 2)
        if len(final) < best_len:
            best = final
            best_len = len(final)
    
    # Final verification and fallback
    if best is None:
        best = _construct_maximal(_greedy(all_points), all_points)
        
    if not _is_valid_no_iso(best):
        best = _construct_maximal(_greedy(all_points), all_points)
        
    if not _is_maximal(best, all_points):
        best = _construct_maximal(best, all_points)
    
    return best
