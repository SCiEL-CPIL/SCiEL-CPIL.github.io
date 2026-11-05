"""Minimum maximal no-isosceles set construction."""
N = 16
import random
import time
import itertools
from collections import defaultdict


def _dist_sq(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        d_pa = _dist_sq(p, a)
        for j in range(i + 1, n):
            b = chosen[j]
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False


def _is_maximal(chosen):
    """Check if chosen is a maximal no-isosceles set."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if not _creates_isosceles(p, chosen):
            return False
    return True


def _greedy_add_to_maximal(chosen, deadline):
    """Greedily add points to make chosen maximal, maintaining no-isosceles."""
    chosen_set = set(chosen)
    # Try adding points in a deterministic order for consistency
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen


def _shrink_once(chosen, deadline):
    """Try to shrink chosen by removing one point and restoring maximality."""
    best = list(chosen)
    best_size = len(best)
    
    # Try removing each point
    for idx in range(len(best)):
        if time.time() > deadline:
            break
        
        candidate = [p for i, p in enumerate(best) if i != idx]
        
        # Restore maximality by adding points
        candidate = _greedy_add_to_maximal(candidate, deadline)
        
        if len(candidate) < best_size and _is_maximal(candidate):
            best = candidate
            best_size = len(best)
    
    return best


def _local_search_shrink(chosen, deadline, rng):
    """Iterative shrinking with random restarts."""
    best = list(chosen)
    best_size = len(best)
    
    while time.time() < deadline:
        # Try multiple random removals
        improved = False
        for _ in range(10):
            if time.time() > deadline:
                break
            
            # Pick a random point to remove
            idx = rng.randint(0, len(best) - 1)
            candidate = [p for i, p in enumerate(best) if i != idx]
            
            # Restore maximality
            candidate = _greedy_add_to_maximal(candidate, deadline)
            
            if len(candidate) < best_size and _is_maximal(candidate):
                best = candidate
                best_size = len(best)
                improved = True
        
        if not improved:
            break
    
    return best


def _build_initial_maximal(rng):
    """Build an initial maximal no-isosceles set using randomized greedy."""
    all_points = list(itertools.product(range(N), repeat=2))
    rng.shuffle(all_points)
    
    chosen = []
    for p in all_points:
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
    
    # Ensure maximality
    chosen = _greedy_add_to_maximal(chosen, time.time() + 5)
    
    return chosen


def construct():
    deadline = time.time() + 290  # Safety margin before 330s limit
    rng = random.Random(42)
    
    best = None
    best_size = float('inf')
    
    # Phase 1: Build initial solutions with diverse random seeds
    for seed in [42, 123, 7, 999, 2024]:
        if time.time() > deadline - 60:
            break
        
        rng_seed = random.Random(seed)
        initial = _build_initial_maximal(rng_seed)
        
        if _is_maximal(initial):
            if len(initial) < best_size:
                best_size = len(initial)
                best = list(initial)
        
        # Phase 2: Shrink the solution
        if time.time() < deadline - 30:
            shrunk = _shrink_once(initial, deadline - 30)
            if len(shrunk) < best_size:
                best_size = len(shrunk)
                best = list(shrunk)
            
            # Phase 3: Local search with random restarts
            if time.time() < deadline - 15:
                local_best = _local_search_shrink(shrunk, deadline - 15, rng)
                if len(local_best) < best_size:
                    best_size = len(local_best)
                    best = local_best
    
    # If no valid solution found, do a final greedy
    if best is None:
        chosen = []
        for p in itertools.product(range(N), repeat=2):
            if not _creates_isosceles(p, chosen):
                chosen.append(p)
        best = chosen
    
    return best
