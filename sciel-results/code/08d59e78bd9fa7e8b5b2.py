"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

N = 16

# EVOLVE-BLOCK-START
import random


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso_fast(p, points, dists):
    """
    Check if p can be added to points without forming an isosceles triangle.
    Uses precomputed distances between existing points.
    dists[i][j] = squared distance between points[i] and points[j]
    """
    n = len(points)
    # Compute distances from p to all existing points
    for i in range(n):
        d_pi = _dist2(p, points[i])
        for j in range(i + 1, n):
            d_pj = _dist2(p, points[j])
            d_ij = dists[i][j]
            if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                return False
    return True


def _compute_dists(points):
    """Compute pairwise squared distances."""
    n = len(points)
    dists = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist2(points[i], points[j])
            dists[i][j] = d
            dists[j][i] = d
    return dists


def _greedy_with_order(order, deadline):
    """Greedy construction following a specific order of points."""
    chosen = []
    dists = []  # list of rows of distance matrix
    for p in order:
        if time.time() > deadline:
            break
        # Check if p can be added
        can_add = True
        n = len(chosen)
        for i in range(n):
            d_pi = _dist2(p, chosen[i])
            for j in range(i + 1, n):
                d_pj = _dist2(p, chosen[j])
                d_ij = dists[i][j]
                if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            # Add p
            chosen.append(p)
            # Update dists
            new_row = [0] * (n + 1)
            for i in range(n):
                d = _dist2(p, chosen[i])
                new_row[i] = d
                dists[i].append(d)
            new_row[n] = 0
            dists.append(new_row)
    return chosen


def construct():
    start_time = time.time()
    deadline = start_time + 320  # 320s to be safe within 330s limit
    
    best_set = None
    best_size = N * N + 1  # impossible upper bound
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    # Try multiple randomized greedy constructions with different orderings
    # Use various strategies for ordering
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if trial == 0:
            # Row-major order
            order = list(all_points)
        elif trial == 1:
            # Column-major order
            order = list(reversed(all_points))
        elif trial == 2:
            # Diagonal order
            order = []
            for d in range(2 * N - 1):
                for x in range(N):
                    y = d - x
                    if 0 <= y < N:
                        order.append((x, y))
        elif trial == 3:
            # Reverse diagonal order
            order = []
            for d in range(2 * N - 2, -1, -1):
                for x in range(N):
                    y = d - x
                    if 0 <= y < N:
                        order.append((x, y))
        else:
            # Random shuffled order
            random.seed(trial * 1000 + 42)
            order = all_points[:]
            random.shuffle(order)
        
        chosen = _greedy_with_order(order, deadline)
        
        # Update best
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen[:]
            
            # Early termination if we find a very small set
            if best_size <= 10:
                break
    
    # Try local search / improvement on the best set found
    if best_set is not None and time.time() < deadline:
        # Try to remove points and see if we can get a smaller maximal set
        # Actually, we want minimal maximal, so we should try to find smaller sets
        # Let's try more aggressive random restarts with time remaining
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            extra_trials = int(remaining_time / 2)  # ~2s per trial
            extra_trials = min(extra_trials, 20)
            for extra in range(extra_trials):
                if time.time() > deadline:
                    break
                random.seed(9999 + extra)
                order = all_points[:]
                random.shuffle(order)
                chosen = _greedy_with_order(order, deadline)
                if len(chosen) < best_size:
                    best_size = len(chosen)
                    best_set = chosen[:]
                    if best_size <= 10:
                        break
    
    if best_set is None:
        # Fallback to deterministic greedy
        chosen = _greedy_with_order(list(all_points), deadline)
        best_set = chosen
        best_size = len(chosen)
    
    return best_set
# EVOLVE-BLOCK-END
