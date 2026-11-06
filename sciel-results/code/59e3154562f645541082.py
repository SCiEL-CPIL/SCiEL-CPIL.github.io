import time
import random
import numpy as np

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
dist2 = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        dx = POINTS[i][0] - POINTS[j][0]
        dy = POINTS[i][1] - POINTS[j][1]
        d = dx * dx + dy * dy
        dist2[i, j] = d
        dist2[j, i] = d

# Precompute for each point p, the list of (a,b) pairs that form isosceles with p
# Use numpy for speed: for each p, find all pairs (a,b) where dist(p,a)==dist(p,b) or dist(p,a)==dist(a,b) or dist(p,b)==dist(a,b)
# Store as a list of sets for fast membership

# Alternative: precompute a conflict matrix approach
# For maximality: point p can be added to S iff no (a,b) in iso_pairs[p] has both a,b in S
# We need fast checking. Let's precompute iso_pairs as lists of tuples.

iso_pairs = [set() for _ in range(NUM_POINTS)]

for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        for k in range(j + 1, NUM_POINTS):
            d1 = dist2[i, j]
            d2 = dist2[i, k]
            d3 = dist2[j, k]
            if d1 == d2 or d1 == d3 or d2 == d3:
                iso_pairs[i].add((j, k))
                iso_pairs[j].add((i, k))
                iso_pairs[k].add((i, j))

# Convert to lists for faster iteration
iso_pairs_list = [list(s) for s in iso_pairs]

def can_add(p, S_set):
    """Check if point p can be added to S_set without creating an isosceles triangle."""
    for a, b in iso_pairs_list[p]:
        if a in S_set and b in S_set:
            return False
    return True

def is_maximal_fast(S_set):
    """Check if S_set is maximal using a faster approach."""
    S_arr = np.zeros(NUM_POINTS, dtype=bool)
    for p in S_set:
        S_arr[p] = True
    
    for p in range(NUM_POINTS):
        if S_arr[p]:
            continue
        # Check if p can be added
        blocked = False
        for a, b in iso_pairs_list[p]:
            if S_arr[a] and S_arr[b]:
                blocked = True
                break
        if not blocked:
            return False
    return True

def greedy_maximal(rng, deadline):
    """Build a maximal set via greedy, then minimize."""
    S_set = set()
    points = list(range(NUM_POINTS))
    rng.shuffle(points)
    
    for p in points:
        if can_add(p, S_set):
            S_set.add(p)
    
    return S_set

def minimize_set(S_set, rng, deadline):
    """Aggressively try to remove points while maintaining maximality."""
    S_set = set(S_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        removal_order = list(S_set)
        rng.shuffle(removal_order)
        
        for p in removal_order:
            if time.time() >= deadline:
                break
            
            trial_S = S_set - {p}
            if is_maximal_fast(trial_S):
                S_set = trial_S
                improved = True
                break  # Restart with new set
    
    return S_set

def swap_minimize(S_set, rng, deadline):
    """Try swaps to reduce size."""
    S_set = set(S_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removals first
        for p in list(S_set):
            if time.time() >= deadline:
                break
            trial_S = S_set - {p}
            if is_maximal_fast(trial_S):
                S_set = trial_S
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps
        for p in list(S_set):
            if time.time() >= deadline:
                break
            
            trial_base = S_set - {p}
            # Find candidates that can be added to trial_base
            candidates = []
            for q in range(NUM_POINTS):
                if q not in trial_base and can_add(q, trial_base):
                    candidates.append(q)
            
            for q in candidates:
                if time.time() >= deadline:
                    break
                trial_S = trial_base | {q}
                if is_maximal_fast(trial_S) and len(trial_S) < len(S_set):
                    S_set = trial_S
                    improved = True
                    break
            if improved:
                break
    
    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + 310  # Safety margin
    
    best_set = None
    best_len = float('inf')
    
    restart_count = 0
    while time.time() < deadline - 15:
        current_rng = random.Random(restart_count * 7919 + 12345)
        
        # Greedy construction
        S_set = greedy_maximal(current_rng, deadline - 5)
        
        # Minimize
        if time.time() < deadline - 5:
            S_set = minimize_set(S_set, current_rng, deadline - 5)
        
        # Swap minimize
        if time.time() < deadline - 5:
            S_set = swap_minimize(S_set, current_rng, deadline - 5)
        
        if len(S_set) < best_len:
            best_len = len(S_set)
            best_set = set(S_set)
        
        restart_count += 1
        
        # If we have a good solution, do intensive local search
        if best_len <= 14 and time.time() < deadline - 10:
            aggressive_rng = random.Random(restart_count * 31 + 999)
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for p in list(best_set):
                    if time.time() >= deadline - 10:
                        break
                    trial_S = best_set - {p}
                    if is_maximal_fast(trial_S):
                        best_set = trial_S
                        best_len = len(best_set)
                        improved = True
                        break
    
    if best_set is None:
        return []
    return [POINTS[i] for i in best_set]
