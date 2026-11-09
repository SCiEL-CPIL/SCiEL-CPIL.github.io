import time
import random
import sys

N = 12
MAX_TIME = 55.0  # Safety margin

def compute_threshold_fast(a_map, b_map):
    """
    Computes the shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    size = 1 << N
    start_mask = size - 1
    
    # Precompute transitions for all subsets
    # This is O(2^N * N) which is 4096 * 12 = ~50k ops, very fast.
    trans_a = [0] * size
    trans_b = [0] * size
    
    for mask in range(size):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= (1 << a_map[i])
                mb |= (1 << b_map[i])
        trans_a[mask] = ma
        trans_b[mask] = mb
        
    # BFS
    dist = [-1] * size
    dist[start_mask] = 0
    queue = [start_mask]
    head = 0
    
    while head < len(queue):
        curr = queue[head]
        head += 1
        d = dist[curr]
        
        # Check if single state (power of 2)
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
    # The Černý automaton (or Pawlik variant) achieves the theoretical bound (N-1)^2
    # For N=12, this is 121.
    # Structure:
    # a(i) = (i + 1) % N
    # b(i) = 0 for i > 0, b(0) = N - 1
    
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] * N
    b_map[0] = N - 1
    for i in range(1, N):
        b_map[i] = 0
        
    # Verify and return
    score = compute_threshold_fast(a_map, b_map)
    
    # If for some reason this isn't the best (e.g. if my memory of the bound is wrong 
    # or implementation is flawed), we could do a quick search.
    # But 121 is the known max for N=12.
    
    # Just in case, let's do a very brief local search to see if we can beat 121 (impossible) 
    # or if we made a mistake and it's lower, and try to fix it.
    # But if it is 121, we return it.
    
    if score == (N - 1) ** 2:
        return [a_map, b_map]
        
    # Fallback: If the construction failed to reach the bound, run a quick SA
    # to find a better one.
    start_time = time.time()
    best_a = a_map.copy()
    best_b = b_map.copy()
    best_score = score
    
    cur_a = a_map.copy()
    cur_b = b_map.copy()
    cur_score = score
    
    T = 5.0
    alpha = 0.99
    
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
            
        if new_score > cur_score:
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = cur_a.copy()
                best_b = cur_b.copy()
        else:
            delta = new_score - cur_score
            if delta < 0:
                prob = pow(2.718281828, delta / T)
                if random.random() < prob:
                    cur_score = new_score
                else:
                    cur_a = prev_a
                    cur_b = prev_b
            # If delta >= 0, it's an improvement, handled above? 
            # Wait, if new_score == cur_score, delta=0, prob=1, accept.
            # If new_score > cur_score, handled above.
            # So only delta < 0 is here.
            
        T = max(0.1, T * alpha)
        
    return [best_a, best_b]
