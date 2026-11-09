import time
import random
import math

N = 12
MAX_TIME = 55.0  # Safety margin before 60s kill

def compute_threshold_fast(a_map, b_map):
    """
    Computes the shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    Optimized to compute transitions on-the-fly using bit tricks.
    """
    size = 1 << N
    start_mask = size - 1
    
    # dist array for BFS
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
            
        # Compute next states on the fly
        # For each state i in curr, add a_map[i] and b_map[i] to next sets
        next_a = 0
        next_b = 0
        
        # Iterate through set bits in curr
        m = curr
        while m:
            lsb = m & -m
            i = (lsb.bit_length() - 1)
            next_a |= (1 << a_map[i])
            next_b |= (1 << b_map[i])
            m ^= lsb
            
        if dist[next_a] == -1:
            dist[next_a] = d + 1
            queue.append(next_a)
            
        if dist[next_b] == -1:
            dist[next_b] = d + 1
            queue.append(next_b)
            
    return -1

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Start with Pawlik automaton
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] + [i - 1 for i in range(1, N)]
    
    best_a = a_map.copy()
    best_b = b_map.copy()
    best_score = compute_threshold_fast(best_a, best_b)
    
    # If best_score is -1, we need a valid one. Pawlik is always valid for N>=2.
    if best_score == -1:
         # Fallback to a known simple synchronizing automaton if Pawlik fails (unlikely)
         a_map = [i+1 for i in range(N-1)] + [0]
         b_map = [0]*N
         best_a = a_map
         best_b = b_map
         best_score = compute_threshold_fast(best_a, best_b)

    # Greedy Hill Climbing with Random Restarts
    # We will run multiple independent restarts until time runs out.
    
    while time.time() - start_time < MAX_TIME:
        # Initialize current state
        if random.random() < 0.1:
            # Occasionally start from best known to refine
            cur_a = best_a.copy()
            cur_b = best_b.copy()
        else:
            # Random restart
            # Bias towards permutations or simple structures? 
            # For now, uniform random is okay but might be slow to find sync.
            # Let's try: a_map is a permutation, b_map is random.
            # Or just fully random.
            cur_a = [random.randint(0, N-1) for _ in range(N)]
            cur_b = [random.randint(0, N-1) for _ in range(N)]
        
        cur_score = compute_threshold_fast(cur_a, cur_b)
        
        # If not synchronizing, try a few quick fixes or restart
        if cur_score == -1:
            # Try to fix by making b_map a constant map (always syncs in 1 step if a is perm? No)
            # Actually, if b_map is constant, it syncs in 1 step. 
            # Let's just restart if not sync.
            continue
        
        # Local Search Phase
        # Limit steps per restart to avoid getting stuck in one local opt for too long
        max_steps = 200
        steps = 0
        while steps < max_steps and time.time() - start_time < MAX_TIME:
            steps += 1
            prev_a = cur_a.copy()
            prev_b = cur_b.copy()
            
            # Mutate: change one transition
            if random.random() < 0.5:
                i = random.randint(0, N - 1)
                cur_a[i] = random.randint(0, N - 1)
            else:
                i = random.randint(0, N - 1)
                cur_b[i] = random.randint(0, N - 1)
                
            new_score = compute_threshold_fast(cur_a, cur_b)
            
            if new_score == -1:
                # Reject non-synchronizing
                cur_a = prev_a
                cur_b = prev_b
                continue
            
            # Accept if better, or with small probability if worse (simulated annealing lite)
            if new_score > cur_score:
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_a = cur_a.copy()
                    best_b = cur_b.copy()
            else:
                # Probability to accept worse move to escape local minima
                # Use a simple heuristic: accept if random < 0.1 (10% chance)
                if random.random() > 0.1:
                    cur_a = prev_a
                    cur_b = prev_b
    
    return [best_a, best_b]
