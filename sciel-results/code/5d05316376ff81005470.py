"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import itertools
import math

DEADLINE = time.time() + 295  # Safety margin before 330s kill

# Precompute all points and squared distances
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_PTS = N * N

# dist[i][j] = squared distance between point i and point j
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d


def _is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle."""
    d1 = dist[i][j]
    d2 = dist[i][k]
    d3 = dist[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p_idx, chosen_set, chosen_indices):
    """Check if point p_idx can be added to the chosen set without creating isosceles triangles."""
    for i, j in itertools.combinations(chosen_indices, 2):
        if _is_isosceles(p_idx, i, j):
            return False
    return True


def _is_maximal(chosen_indices):
    """Check if the set is maximal (no further point can be added)."""
    chosen_set = set(chosen_indices)
    for p in range(NUM_PTS):
        if p not in chosen_set:
            if _can_add(p, chosen_set, chosen_indices):
                return False
    return True


def _greedy_randomized(seed=None):
    """Greedy randomized construction."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    order = list(range(NUM_PTS))
    rng.shuffle(order)
    
    chosen = []
    chosen_set = set()
    
    for p in order:
        if p in chosen_set:
            continue
        # Check if p can be added
        can = True
        for i, j in itertools.combinations(chosen, 2):
            if _is_isosceles(p, i, j):
                can = False
                break
        if can:
            chosen.append(p)
            chosen_set.add(p)
    
    return chosen


def _local_search(chosen, max_iters=50):
    """Local search: try removing points and refilling greedily."""
    best = list(chosen)
    best_len = len(best)
    
    for iteration in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Try removing each point and see if we can get a smaller maximal set
        for idx in range(len(best)):
            candidate = [x for i, x in enumerate(best) if i != idx]
            candidate_set = set(candidate)
            
            # Try to greedily add points to make it maximal, but try to keep it small
            # Actually, for minimization, we want to see if removing a point still allows
            # a valid (possibly non-maximal) set, then check if it can be made maximal
            # with fewer additions than we removed.
            
            # Simpler: just check if the set without this point is still valid (no isosceles)
            # It always is since we're removing. Then check if it's maximal.
            # If not maximal, we need to add points. The question is whether the final
            # maximal set is smaller.
            
            # For now, let's just try: remove point, then greedily fill to maximal
            # and see if result is smaller.
            
            fill_order = list(range(NUM_PTS))
            random.shuffle(fill_order)
            
            new_chosen = list(candidate)
            new_set = set(candidate)
            
            for p in fill_order:
                if p in new_set:
                    continue
                can = True
                for i, j in itertools.combinations(new_chosen, 2):
                    if _is_isosceles(p, i, j):
                        can = False
                        break
                if can:
                    new_chosen.append(p)
                    new_set.add(p)
            
            if len(new_chosen) < best_len:
                best = new_chosen
                best_len = len(best)
                improved = True
                break
        
        if not improved:
            break
    
    return best


def construct():
    """Construct a minimal maximal no-isosceles set."""
    best = None
    best_len = float('inf')
    
    # Multiple randomized greedy restarts
    num_restarts = 0
    while time.time() < DEADLINE - 10:
        chosen = _greedy_randomized(seed=num_restarts)
        if len(chosen) < best_len:
            best = chosen
            best_len = len(chosen)
        
        # Try local search on the best found so far
        if len(chosen) <= best_len:
            improved = _local_search(chosen, max_iters=20)
            if len(improved) < best_len:
                best = improved
                best_len = len(best)
        
        num_restarts += 1
    
    if best is None:
        # Fallback: simple greedy
        best = _greedy_randomized(seed=0)
    
    return [POINTS[i] for i in best]
# EVOLVE-BLOCK-END
