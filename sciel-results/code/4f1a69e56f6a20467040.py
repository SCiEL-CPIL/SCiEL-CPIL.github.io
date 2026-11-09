"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

N = 16

# EVOLVE-BLOCK-START
import random
import math


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _can_add(p, chosen, dists):
    """Check if point p can be added to chosen without creating isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        d_pi = _dist2(p, chosen[i])
        for j in range(i + 1, n):
            d_pj = _dist2(p, chosen[j])
            d_ij = dists[i][j]
            if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                return False
    return True


def _add_point(p, chosen, dists):
    """Add point p to chosen and update dists matrix."""
    n = len(chosen)
    new_row = [0] * (n + 1)
    for i in range(n):
        d = _dist2(p, chosen[i])
        new_row[i] = d
        dists[i].append(d)
    new_row[n] = 0
    dists.append(new_row)
    chosen.append(p)


def _is_maximal(chosen, all_points, dists):
    """Check if chosen is maximal - no other point can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen, dists):
            return False
    return True


def _greedy_with_order(order, deadline):
    """Greedy construction following a specific order of points."""
    chosen = []
    dists = []
    for p in order:
        if time.time() > deadline:
            break
        if _can_add(p, chosen, dists):
            _add_point(p, chosen, dists)
    return chosen, dists


def _local_search_improve(chosen, dists, all_points, deadline):
    """
    Try to find a smaller maximal set by removing points and rebuilding.
    For each point in chosen, try removing it and see if we can still
    reach a maximal set with fewer points.
    """
    best = (len(chosen), chosen[:], [row[:] for row in dists])
    
    # Try removing each point one at a time
    n = len(chosen)
    for idx in range(n):
        if time.time() > deadline:
            break
        
        # Create a copy without point idx
        new_chosen = chosen[:idx] + chosen[idx+1:]
        new_dists = []
        for i, row in enumerate(dists):
            if i == idx:
                continue
            new_row = row[:idx] + row[idx+1:]
            new_dists.append(new_row)
        
        # Now try to greedily add points to new_chosen to make it maximal
        # But we want a smaller set, so we should be selective
        # Use a greedy approach but stop when maximal
        all_points_set = set(new_chosen)
        remaining = [p for p in all_points if p not in all_points_set]
        
        # Shuffle remaining for randomness
        random.shuffle(remaining)
        
        improved = False
        for p in remaining:
            if time.time() > deadline:
                break
            if _can_add(p, new_chosen, new_dists):
                _add_point(p, new_chosen, new_dists)
                if len(new_chosen) < len(best[1]):
                    best = (len(new_chosen), new_chosen[:], [row[:] for row in new_dists])
                    improved = True
        
        if not improved and len(new_chosen) < len(best[1]):
            best = (len(new_chosen), new_chosen[:], [row[:] for row in new_dists])
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 320  # 320s to be safe within 330s limit
    
    best_set = None
    best_size = N * N + 1  # impossible upper bound
    best_dists = None
    
    all_points = list(itertools.product(range(N), repeat=2))
    all_points_set = set(all_points)
    
    # Phase 1: Multiple randomized greedy constructions
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if trial == 0:
            order = list(all_points)
        elif trial == 1:
            order = list(reversed(all_points))
        elif trial == 2:
            order = []
            for d in range(2 * N - 1):
                for x in range(N):
                    y = d - x
                    if 0 <= y < N:
                        order.append((x, y))
        elif trial == 3:
            order = []
            for d in range(2 * N - 2, -1, -1):
                for x in range(N):
                    y = d - x
                    if 0 <= y < N:
                        order.append((x, y))
        elif trial == 4:
            # Spiral order from center
            cx, cy = N // 2, N // 2
            order = []
            for r in range(N):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if abs(dx) == r or abs(dy) == r:
                            x, y = cx + dx, cy + dy
                            if 0 <= x < N and 0 <= y < N:
                                order.append((x, y))
        else:
            random.seed(trial * 7919 + 13)
            order = all_points[:]
            random.shuffle(order)
        
        chosen, dists = _greedy_with_order(order, deadline)
        
        # Check if this is a valid maximal set
        if _is_maximal(chosen, all_points, dists):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_set = chosen[:]
                best_dists = [row[:] for row in dists]
                if best_size <= 10:
                    break
    
    # Phase 2: Local search improvement on the best set
    if best_set is not None and time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            # Try multiple rounds of local search
            for round_num in range(5):
                if time.time() > deadline:
                    break
                
                improved_size, improved_set, improved_dists = _local_search_improve(
                    best_set, best_dists, all_points, deadline
                )
                
                if improved_size < best_size:
                    best_size = improved_size
                    best_set = improved_set
                    best_dists = improved_dists
                else:
                    break
    
    if best_set is None:
        # Fallback to deterministic greedy
        chosen, dists = _greedy_with_order(list(all_points), deadline)
        best_set = chosen
        best_size = len(chosen)
    
    return best_set
# EVOLVE-BLOCK-END
