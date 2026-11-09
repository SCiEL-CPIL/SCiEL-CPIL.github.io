"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys

DEADLINE = time.time() + 290  # Safety margin before 330s kill

# Precompute coordinates and squared distances
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_PTS = N * N
coords = POINTS  # index i -> (x, y)

# dist[i][j] = squared distance between point i and point j
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = coords[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = coords[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d


def is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle."""
    d1 = dist[i][j]
    d2 = dist[i][k]
    d3 = dist[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3


def can_add(p, S_indices):
    """Check if point p can be added to set S_indices without creating isosceles."""
    for idx_a in range(len(S_indices)):
        a = S_indices[idx_a]
        for idx_b in range(idx_a + 1, len(S_indices)):
            b = S_indices[idx_b]
            if is_isosceles(p, a, b):
                return False
    return True


def is_maximal(S_indices):
    """Check if S is maximal: no point outside S can be added."""
    S_set = set(S_indices)
    for p in range(NUM_PTS):
        if p not in S_set:
            if can_add(p, S_indices):
                return False
    return True


def greedy_random(seed_val):
    """Greedy construction with random ordering."""
    rng = random.Random(seed_val)
    order = list(range(NUM_PTS))
    rng.shuffle(order)
    chosen = []
    for p in order:
        if can_add(p, chosen):
            chosen.append(p)
    # Ensure maximality by adding any remaining addable points
    S_set = set(chosen)
    for p in range(NUM_PTS):
        if p not in S_set:
            if can_add(p, chosen):
                chosen.append(p)
                S_set.add(p)
    return chosen


def local_improve(S_indices, rng, max_iters=200):
    """Try to reduce the set by removing points and re-filling."""
    S = list(S_indices)
    best = S
    best_len = len(S)
    
    for iteration in range(max_iters):
        if time.time() > DEADLINE:
            break
        if len(S) <= best_len:
            # Try removing a random point
            if len(S) > 1:
                idx = rng.randint(0, len(S) - 1)
                candidate = S[:idx] + S[idx+1:]
                # Check if candidate is still no-isosceles (it should be, subset)
                # Check if we can make it maximal
                S_set = set(candidate)
                fillable = []
                for p in range(NUM_PTS):
                    if p not in S_set:
                        if can_add(p, candidate):
                            fillable.append(p)
                new_S = candidate + fillable
                if len(new_S) < best_len:
                    best = new_S
                    best_len = len(new_S)
                    S = new_S
                elif len(new_S) == best_len and rng.random() < 0.3:
                    # Accept equal-length to explore
                    S = new_S
        else:
            # S grew, reset to best
            S = list(best)
    
    return best


def construct():
    best_solution = None
    best_len = float('inf')
    
    # First, get a baseline from deterministic greedy
    baseline = greedy_random(42)
    best_solution = baseline
    best_len = len(baseline)
    
    # Try multiple random restarts
    seed_counter = 0
    while time.time() < DEADLINE:
        seed_counter += 1
        sol = greedy_random(seed_counter)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
        
        # Try local improvement
        rng = random.Random(seed_counter * 1000 + 7)
        improved = local_improve(sol, rng, max_iters=50)
        if len(improved) < best_len:
            best_solution = improved
            best_len = len(improved)
        
        # Check deadline
        if time.time() > DEADLINE - 30:
            break
    
    # Final check: ensure the best solution is valid and maximal
    if best_solution is None:
        # Fallback to simple greedy
        chosen = []
        for p in range(NUM_PTS):
            if can_add(p, chosen):
                chosen.append(p)
        best_solution = chosen
    
    # Convert indices to coordinates
    return [coords[i] for i in best_solution]

# EVOLVE-BLOCK-END
