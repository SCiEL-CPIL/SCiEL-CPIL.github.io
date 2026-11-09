import time
import random
import sys

N = 12
MAX_TIME = 50.0  # Safety margin

def compute_threshold(a_map, b_map):
    """
    Computes the shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    # State: bitmask of active states
    # Initial state: all states active
    start_mask = (1 << N) - 1
    target_mask = 0  # Actually, any single state mask is a target.
    # We want to reach any mask with exactly one bit set.
    
    # Precompute transitions for all possible subsets? No, too many (2^12 = 4096 is small).
    # We can precompute next_state for each subset and each letter.
    
    # dest[subset][letter]
    # But computing on the fly is fine if we cache.
    
    visited = {start_mask: 0}
    queue = [start_mask]
    dist = {start_mask: 0}
    
    # Precompute transition of a mask by 'a' and 'b'
    # mask_a = union of a_map[i] for i in mask
    # mask_b = union of b_map[i] for i in mask
    
    # Optimization: Precompute transition tables for all 4096 masks?
    # 4096 * 2 * 12 operations is small. Let's do it for speed.
    
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= (1 << a_map[i])
                mb |= (1 << b_map[i])
        trans_a[mask] = ma
        trans_b[mask] = mb
    
    # Check if start is already a single state
    if start_mask == 0 or (start_mask & (start_mask - 1)) == 0:
        return 0
        
    while queue:
        current = queue.pop(0)
        d = dist[current]
        
        # Check if current is a single state
        if (current & (current - 1)) == 0:
            return d
            
        # Expand neighbors
        next_a = trans_a[current]
        next_b = trans_b[current]
        
        if next_a not in dist:
            dist[next_a] = d + 1
            queue.append(next_a)
            
        if next_b not in dist:
            dist[next_b] = d + 1
            queue.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    """
    Uses Simulated Annealing to find a high-threshold synchronizing automaton.
    """
    start_time = time.time()
    
    # Initialize with a known good structure or random
    # Let's start with the "Pawlik" structure which is at least synchronizing
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] + [i - 1 for i in range(1, N)]
    
    best_score = compute_threshold(a_map, b_map)
    best_a = a_map.copy()
    best_b = b_map.copy()
    
    current_a = a_map.copy()
    current_b = b_map.copy()
    current_score = best_score
    
    temperature = 10.0
    cooling = 0.999
    
    while time.time() - start_time < MAX_TIME:
        # Mutation
        # Change a random transition in a or b
        if random.random() < 0.5:
            i = random.randint(0, N - 1)
            current_a[i] = random.randint(0, N - 1)
        else:
            i = random.randint(0, N - 1)
            current_b[i] = random.randint(0, N - 1)
            
        new_score = compute_threshold(current_a, current_b)
        
        # If not synchronizing, treat score as very low
        if new_score == -1:
            new_score = -1000
            
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = current_a.copy()
                best_b = current_b.copy()
        else:
            # Accept with probability
            delta = new_score - current_score
            # delta is negative
            prob = random.expovariate(1.0 / temperature)
            if prob < -delta: # Or exp(delta/temp)
                current_score = new_score
            else:
                # Revert mutation
                if random.random() < 0.5: # We don't know which one was mutated, so we can't easily revert without storing prev
                    # This implementation is flawed for reverting. 
                    # Better to store prev_a, prev_b
                    pass
        
        # Simple approach: Just keep the current state if accepted, else revert.
        # To revert correctly, we need to store the previous state.
        # Let's restart the loop logic slightly for clarity and correctness.
        pass
        
    # The above loop is messy. Let's implement a cleaner SA.
    
    # Clean SA Implementation
    start_time = time.time()
    
    # Start with random or structured
    # Structured start is better
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] + [i - 1 for i in range(1, N)]
    
    best_a = a_map.copy()
    best_b = b_map.copy()
    best_score = compute_threshold(best_a, best_b)
    
    cur_a = a_map.copy()
    cur_b = b_map.copy()
    cur_score = best_score
    
    T = 5.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() - start_time < MAX_TIME:
        # Store previous
        prev_a = cur_a.copy()
        prev_b = cur_b.copy()
        
        # Mutate
        if random.random() < 0.5:
            i = random.randint(0, N - 1)
            cur_a[i] = random.randint(0, N - 1)
        else:
            i = random.randint(0, N - 1)
            cur_b[i] = random.randint(0, N - 1)
            
        new_score = compute_threshold(cur_a, cur_b)
        if new_score == -1:
            new_score = -1000
            
        if new_score > cur_score:
            # Accept
            if new_score > best_score:
                best_score = new_score
                best_a = cur_a.copy()
                best_b = cur_b.copy()
        else:
            # Metropolis
            delta = new_score - cur_score
            if delta < 0:
                prob = pow(2.718281828, delta / T)
                if random.random() < prob:
                    # Accept
                    pass
                else:
                    # Revert
                    cur_a = prev_a
                    cur_b = prev_b
                    new_score = cur_score # Score doesn't change if reverted
            
        # Update current score if accepted
        # Note: If reverted, cur_score remains same. If accepted, cur_score becomes new_score.
        # The logic above updates best, but we need to update cur_score correctly.
        
        # Let's refine the acceptance logic
        if new_score > cur_score or (new_score <= cur_score and random.random() < pow(2.718281828, (new_score - cur_score) / T)):
             cur_score = new_score
             if new_score > best_score:
                 best_score = new_score
                 best_a = cur_a.copy()
                 best_b = cur_b.copy()
        else:
             cur_a = prev_a
             cur_b = prev_b
             # cur_score remains unchanged
            
        T = max(T_min, T * alpha)
        
    return [best_a, best_b]

# The above code has a flaw: it re-runs compute_threshold which is O(2^N * N).
# 4096 * 12 * 2 = ~100k ops per eval.
# In 50s, Python can do maybe 1M-10M ops?
# 100k ops/eval -> 10-100 evals/sec.
# 50s -> 500-5000 evals.
# This is likely too slow to explore the space effectively with pure SA from a trivial start.
# We need a faster threshold calculation or a smarter search.

# Optimization:
# The BFS in compute_threshold is the bottleneck.
# We can precompute the transition tables for the CURRENT automaton once,
# but they change every step.
# However, we can speed up the BFS by using a deque and checking single-bit masks early.
# Also, we can limit the BFS depth. If we find a reset word of length L, we don't need to go deeper.
# But we need the EXACT length.

# Alternative: Use a heuristic that doesn't require exact BFS for every step?
# No, the score is the exact threshold.

# Let's optimize compute_threshold.
# 1. Use a list for queue (BFS) instead of pop(0).
# 2. Use arrays for visited/dist instead of dicts.

def compute_threshold_fast(a_map, b_map):
    size = 1 << N
    trans_a = [0] * size
    trans_b = [0] * size
    
    # Precompute transitions
    for mask in range(size):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= (1 << a_map[i])
                mb |= (1 << b_map[i])
        trans_a[mask] = ma
        trans_b[mask] = mb
        
    start_mask = size - 1
    dist = [-1] * size
    dist[start_mask] = 0
    queue = [start_mask]
    head = 0
    
    while head < len(queue):
        curr = queue[head]
        head += 1
        d = dist[curr]
        
        # Check if single state
        if curr == 0 or (curr & (curr - 1)) == 0:
            return d
            
        next_a = trans_a[curr]
        if dist[next_a] == -1:
            dist[next_a] = d + 1
            queue.append(next_a)
            
        next_b = trans_b[curr]
        if dist[next_b] == -1:
            dist[next_b] = d + 1
            queue.append(next_b)
            
    return -1

def construct():
    start_time = time.time()
    random.seed(42) # For reproducibility if needed, or just let it run
    
    # Start with a random automaton that is likely synchronizing?
    # Or start with the Pawlik one.
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] + [i - 1 for i in range(1, N)]
    
    best_a = a_map.copy()
    best_b = b_map.copy()
    best_score = compute_threshold_fast(best_a, best_b)
    
    cur_a = a_map.copy()
    cur_b = b_map.copy()
    cur_score = best_score
    
    T = 10.0
    T_min = 0.1
    alpha = 0.9995
    
    while time.time() - start_time < MAX_TIME:
        prev_a = cur_a.copy()
        prev_b = cur_b.copy()
        
        # Mutate
        if random.random() < 0.5:
            i = random.randint(0, N - 1)
            cur_a[i] = random.randint(0, N - 1)
        else:
            i = random.randint(0, N - 1)
            cur_b[i] = random.randint(0, N - 1)
            
        new_score = compute_threshold_fast(cur_a, cur_b)
        if new_score == -1:
            new_score = -1000
            
        # Acceptance
        accept = False
        if new_score > cur_score:
            accept = True
        else:
            delta = new_score - cur_score
            if delta < 0:
                # Use exp(delta/T)
                # To avoid underflow/overflow, compare log(prob) < log(random)
                # log(exp(delta/T)) = delta/T
                # log(random) is negative
                # If delta/T > log(random), accept
                # Actually, standard is: if random() < exp(delta/T)
                # If delta is large negative, exp is 0.
                try:
                    prob = pow(2.718281828, delta / T)
                    if random.random() < prob:
                        accept = True
                except:
                    accept = False
        
        if accept:
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = cur_a.copy()
                best_b = cur_b.copy()
        else:
            cur_a = prev_a
            cur_b = prev_b
            
        T = max(T_min, T * alpha)
        
    return [best_a, best_b]
