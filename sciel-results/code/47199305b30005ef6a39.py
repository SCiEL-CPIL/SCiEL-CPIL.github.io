"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 16
_TIME_LIMIT = 320  # safety margin before 330s kill

# EVOLVE-BLOCK-START
import random
import numpy as np


def _dist_sq(a, b):
    """Squared distance between two grid points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _is_isosceles(p, a, b):
    """Check if triangle (p, a, b) is isosceles (any two sides equal)."""
    d_pa = _dist_sq(p, a)
    d_pb = _dist_sq(p, b)
    d_ab = _dist_sq(a, b)
    return d_pa == d_pb or d_pa == d_ab or d_pb == d_ab


def _can_add(p, chosen):
    """Check if point p can be added without creating an isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_isosceles(p, chosen[i], chosen[j]):
                return False
    return True


def _greedy_order(order, deadline):
    """Greedy construction: add point if it doesn't create isosceles."""
    chosen = []
    for p in order:
        if time.time() > deadline:
            break
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _try_remove_then_add(chosen, p, deadline):
    """Try to add p by removing one or two existing points.
    Returns (new_chosen, improvement) where improvement = len(chosen) - len(new_chosen)."""
    # Try removing 0 points (just add)
    if _can_add(p, chosen):
        return chosen + [p], 1  # improvement of 1 (we added one)
    
    # Try removing 1 point
    for i in range(len(chosen)):
        candidate = chosen[:i] + chosen[i+1:]
        if _can_add(p, candidate):
            return candidate + [p], 1  # same size, but different set (might help later)
    
    # Try removing 2 points
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            candidate = chosen[:i] + chosen[i+1:j] + chosen[j+1:]
            if _can_add(p, candidate):
                return candidate + [p], 0  # reduced by 1
    
    return None, 0


def _local_search(chosen, deadline):
    """Local search: try to reduce set size by removing points."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(chosen)
        best_size = current_size
        best_chosen = chosen
        
        for i in range(current_size):
            candidate = chosen[:i] + chosen[i+1:]
            # Check if the set is still maximal (no point can be added)
            # For now, just check if it's still valid (no isosceles) - removing preserves validity
            # But we need it to be maximal. Let's check maximality.
            if _is_maximal(candidate, deadline):
                if len(candidate) < best_size:
                    best_size = len(candidate)
                    best_chosen = candidate
                    improved = True
                    break
    return best_chosen


def _is_maximal(chosen, deadline):
    """Check if no grid point can be added to chosen without creating isosceles."""
    chosen_set = set(chosen)
    for x in range(N):
        for y in range(N):
            if (x, y) in chosen_set:
                continue
            p = (x, y)
            if _can_add(p, chosen):
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT - 10  # 10s safety margin
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Multiple random restarts with different orderings
    best = None
    best_size = float('inf')
    
    # First try a few deterministic orderings
    orderings = [
        list(all_points),  # row-major
        list(reversed(all_points)),  # reverse row-major
        sorted(all_points, key=lambda p: (p[0] + p[1], p[0])),  # diagonal
        sorted(all_points, key=lambda p: (p[0] - p[1], p[0])),  # anti-diagonal
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        chosen = _greedy_order(order, deadline)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen
    
    # Random restarts
    while time.time() < deadline:
        random.shuffle(all_points)
        chosen = _greedy_order(all_points, deadline)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen
    
    if best is None:
        best = _greedy_order(all_points, deadline)
    
    return best
# EVOLVE-BLOCK-END
