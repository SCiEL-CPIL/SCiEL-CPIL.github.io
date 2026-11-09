"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
ALL_POINTS_SET = set(ALL_POINTS)


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(p, chosen):
    """Check if p forms an isosceles triangle with any pair in chosen."""
    n = len(chosen)
    for i in range(n):
        pi = chosen[i]
        d_pi = _d2(p, pi)
        for j in range(i + 1, n):
            pj = chosen[j]
            d_pj = _d2(p, pj)
            d_ij = _d2(pi, pj)
            if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                return True
    return False


def _can_add(p, chosen):
    """Check if p can be added to chosen without creating an isosceles triangle."""
    return not _is_iso(p, chosen)


def _is_maximal(chosen):
    """Check if no further grid point can be added to chosen."""
    chosen_set = set(chosen)
    for p in ALL_POINTS:
        if p not in chosen_set and _can_add(p, chosen):
            return False
    return True


def _greedy(order):
    """Greedy construction using a given ordering of points."""
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _try_remove(chosen, idx):
    """Try removing point at idx; return new list if still maximal, else None."""
    new_chosen = chosen[:idx] + chosen[idx+1:]
    if _is_maximal(new_chosen):
        return new_chosen
    return None


def _local_search_remove(chosen, deadline):
    """Try to remove points one by one until no more can be removed."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(chosen)):
            new_chosen = _try_remove(chosen, i)
            if new_chosen is not None:
                chosen = new_chosen
                improved = True
                break  # restart scan from beginning
    return chosen


def _swap_search(chosen, deadline):
    """Try swapping one point for another to reduce size or maintain size with better structure."""
    best = list(chosen)
    for _ in range(200):  # limited swaps per call
        if time.time() >= deadline:
            break
        if len(best) <= 10:
            break
        # Try removing a random point and adding a random other point
        i = random.randint(0, len(best) - 1)
        removed = best[i]
        candidate = best[:i] + best[i+1:]
        
        # Try to add a point that's not in candidate and not removed
        available = [p for p in ALL_POINTS if p not in candidate and p != removed]
        random.shuffle(available)
        
        found = False
        for p in available[:50]:  # limit candidates per swap attempt
            if _can_add(p, candidate):
                new_set = candidate + [p]
                if _is_maximal(new_set):
                    best = new_set
                    found = True
                    break
        if not found:
            # Try just removing
            if _is_maximal(candidate):
                best = candidate
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 280  # safety margin
    
    # Generate multiple greedy orderings and pick the best initial set
    best_set = None
    best_size = float('inf')
    
    # Row-major
    order = list(itertools.product(range(N), repeat=2))
    s = _greedy(order)
    if len(s) < best_size:
        best_size = len(s)
        best_set = list(s)
    
    # Column-major
    order = list(itertools.product(range(N), repeat=2))
    order.sort(key=lambda p: (p[1], p[0]))
    s = _greedy(order)
    if len(s) < best_size:
        best_size = len(s)
        best_set = list(s)
    
    # Diagonal ordering
    order = list(itertools.product(range(N), repeat=2))
    order.sort(key=lambda p: (p[0] + p[1], p[0]))
    s = _greedy(order)
    if len(s) < best_size:
        best_size = len(s)
        best_set = list(s)
    
    # Checkerboard
    order = list(itertools.product(range(N), repeat=2))
    order.sort(key=lambda p: ((p[0] + p[1]) % 2, p[0], p[1]))
    s = _greedy(order)
    if len(s) < best_size:
        best_size = len(s)
        best_set = list(s)
    
    # Random orderings
    for _ in range(20):
        if time.time() >= deadline:
            break
        order = list(ALL_POINTS)
        random.shuffle(order)
        s = _greedy(order)
        if len(s) < best_size:
            best_size = len(s)
            best_set = list(s)
    
    # Local search: try to remove points
    best_set = _local_search_remove(best_set, deadline)
    
    # More random restarts with local search
    for _ in range(50):
        if time.time() >= deadline:
            break
        order = list(ALL_POINTS)
        random.shuffle(order)
        s = _greedy(order)
        s = _local_search_remove(s, deadline)
        if len(s) < best_size:
            best_size = len(s)
            best_set = list(s)
    
    # Swap search on the best set
    if time.time() < deadline:
        best_set = _swap_search(best_set, deadline)
    
    # Final verification
    if not _is_maximal(best_set):
        # Fallback: rebuild
        order = list(itertools.product(range(N), repeat=2))
        best_set = _greedy(order)
    
    return best_set
