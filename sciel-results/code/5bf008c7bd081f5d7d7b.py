"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def _precompute_iso_conflicts(all_points):
    """
    For each pair of points (i, j), compute the set of point indices k
    such that points[i], points[j], points[k] form an isosceles triangle.
    Returns a list of sets: conflict[i][j] = set of k indices.
    """
    n = len(all_points)
    pts = np.array(all_points, dtype=np.int32)
    # Precompute squared distances
    # dist[i,j] = squared distance between points i and j
    dx = pts[:, 0:1] - pts[1:0, 0]  # won't work directly, let's do it properly
    # Use broadcasting
    diff = pts[:, None, :] - pts[None, :, :]  # shape (n, n, 2)
    sq_dist = np.sum(diff ** 2, axis=2)  # shape (n, n)
    
    conflict = [None] * n
    for i in range(n):
        conflict[i] = [None] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            # For each k, check if (i,j,k) is isosceles
            # d_ij = sq_dist[i][j], d_ik = sq_dist[i][k], d_jk = sq_dist[j][k]
            d_ij = sq_dist[i, j]
            d_ik = sq_dist[i, :]
            d_jk = sq_dist[j, :]
            # Isosceles if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk
            iso_mask = (d_ij == d_ik) | (d_ij == d_jk) | (d_ik == d_jk)
            iso_mask[i] = False  # exclude self
            iso_mask[j] = False  # exclude self
            conflict[i][j] = np.where(iso_mask)[0]
            conflict[j][i] = conflict[i][j]
    
    return conflict, sq_dist


def _greedy_with_conflicts(order, conflict, all_points):
    """
    Greedy construction using precomputed conflict sets.
    A point p can be added if for no pair (a,b) in chosen does p appear in conflict[a][b].
    """
    chosen = []
    chosen_indices = []
    # For each candidate point, track how many chosen pairs block it
    # Actually, simpler: for each new point, check against all pairs in chosen
    # But we can optimize: maintain for each point the count of blocking pairs
    
    n = len(all_points)
    block_count = np.zeros(n, dtype=np.int32)
    
    for p_idx in order:
        # Check if p_idx is blocked by any pair in chosen
        # For each pair (a, b) in chosen, check if p_idx in conflict[a][b]
        ok = True
        for i in range(len(chosen_indices)):
            for j in range(i + 1, len(chosen_indices)):
                a = chosen_indices[i]
                b = chosen_indices[j]
                if p_idx in conflict[a][b]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(p_idx)
            chosen_indices.append(p_idx)
    
    return chosen


def _greedy_priority(order, conflict, all_points):
    """
    Greedy with priority: at each step, among all addable points, pick the one
    that blocks the most other points (to reduce future options and potentially
    lead to smaller maximal sets).
    """
    n = len(all_points)
    chosen_indices = []
    remaining = set(order)
    
    while remaining:
        # Find all addable points
        addable = []
        for p_idx in remaining:
            ok = True
            for i in range(len(chosen_indices)):
                for j in range(i + 1, len(chosen_indices)):
                    a = chosen_indices[i]
                    b = chosen_indices[j]
                    if p_idx in conflict[a][b]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                addable.append(p_idx)
        
        if not addable:
            break
        
        # Pick the point that blocks the most other remaining points
        best_p = addable[0]
        best_block = -1
        for p_idx in addable:
            # Count how many remaining points would be blocked by adding p_idx
            block = 0
            for q_idx in remaining:
                if q_idx == p_idx:
                    continue
                # Check if p_idx and q_idx form a pair that blocks something
                # Actually, we want to know: after adding p_idx, how many new points become blocked?
                # This is complex. Simpler heuristic: pick randomly from addable
                pass
            # For now, just pick randomly
            break
        best_p = random.choice(addable)
        chosen_indices.append(best_p)
        remaining.discard(best_p)
    
    return chosen_indices


def _is_maximal_fast(chosen_indices, conflict, all_points):
    """Check maximality: no point outside chosen can be added."""
    chosen_set = set(chosen_indices)
    n = len(all_points)
    for p_idx in range(n):
        if p_idx in chosen_set:
            continue
        # Check if p_idx can be added
        ok = True
        for i in range(len(chosen_indices)):
            for j in range(i + 1, len(chosen_indices)):
                a = chosen_indices[i]
                b = chosen_indices[j]
                if p_idx in conflict[a][b]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return False  # p_idx can be added, not maximal
    return True


def _local_search(chosen_indices, conflict, all_points, deadline):
    """
    Local search: try to swap points to reduce size while maintaining no-isosceles and maximality.
    Actually, we can't reduce size directly. Instead, try different orderings.
    """
    # This is a placeholder for more sophisticated local search
    return chosen_indices


def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    start = time.time()
    deadline = start + 280  # safety margin before 330s kill
    
    # Precompute conflict sets
    conflict, sq_dist = _precompute_iso_conflicts(all_points)
    
    best = None
    best_size = n_points + 1
    
    random.seed(42)
    
    # Strategy 1: Random greedy with different seeds
    for attempt in range(200):
        if time.time() > deadline - 60:
            break
        order = list(range(n_points))
        random.shuffle(order)
        chosen = _greedy_with_conflicts(order, conflict, all_points)
        if _is_maximal_fast(chosen, conflict, all_points):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen
    
    # Strategy 2: Try sorted orders (row-major, column-major, diagonal, etc.)
    if time.time() < deadline - 40:
        orders = [
            list(range(n_points)),  # row-major
            sorted(range(n_points), key=lambda i: (all_points[i][1], all_points[i][0])),  # column-major
            sorted(range(n_points), key=lambda i: (all_points[i][0] + all_points[i][1], all_points[i][0])),  # diagonal
            sorted(range(n_points), key=lambda i: abs(all_points[i][0] - all_points[i][1])),  # anti-diagonal
        ]
        for order in orders:
            if time.time() > deadline - 40:
                break
            chosen = _greedy_with_conflicts(order, conflict, all_points)
            if _is_maximal_fast(chosen, conflict, all_points):
                if len(chosen) < best_size:
                    best_size = len(chosen)
                    best = chosen
    
    # Strategy 3: More random attempts with time remaining
    while time.time() < deadline - 5:
        order = list(range(n_points))
        random.shuffle(order)
        chosen = _greedy_with_conflicts(order, conflict, all_points)
        if _is_maximal_fast(chosen, conflict, all_points):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen
    
    if best is None:
        # Fallback
        order = list(range(n_points))
        best = _greedy_with_conflicts(order, conflict, all_points)
    
    return [all_points[i] for i in best]
# EVOLVE-BLOCK-END
