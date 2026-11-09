import time
import random
import sys

N = 12
MAX_TIME = 85.0  # Safety margin before 90s kill

def compute_threshold_fast(a_map, b_map):
    """
    Computes the shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    size = 1 << N
    start_mask = size - 1
    
    # Precompute transitions for all masks
    # trans_a[mask] = union of a_map[i] for i in mask
    # trans_b[mask] = union of b_map[i] for i in mask
    trans_a = [0] * size
    trans_b = [0] * size
    
    # Optimization: Use DP to compute transitions faster?
    # trans_a[0] = 0
    # trans_a[mask] = trans_a[mask without lsb] | (1 << a_map[lsb_index])
    
    for mask in range(1, size):
        # Get least significant bit
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        trans_a[mask] = trans_a[prev_mask] | (1 << a_map[idx])
        trans_b[mask] = trans_b[prev_mask] | (1 << b_map[idx])
        
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
        if curr and (curr & (curr - 1)) == 0:
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
    random.seed(42)
    
    # Initial structure: Pawlik automaton
    # a: (i+1) % N
    # b: 0 if i==0 else i-1
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] + [i - 1 for i in range(1, N)]
    
    best_a = a_map.copy()
    best_b = b_map.copy()
    best_score = compute_threshold_fast(best_a, best_b)
    
    cur_a = a_map.copy()
    cur_b = b_map.copy()
    cur_score = best_score
    
    # Simulated Annealing parameters
    T = 15.0
    T_min = 0.1
    alpha = 0.9998
    
    while time.time() - start_time < MAX_TIME:
        # Check time more frequently to avoid overshooting
        if time.time() - start_time > MAX_TIME:
            break
            
        prev_a = cur_a.copy()
        prev_b = cur_b.copy()
        
        # Mutation strategy:
        # 1. Change one random transition (most common)
        # 2. Swap destinations of two states in a or b
        # 3. Change two random transitions
        
        r = random.random()
        if r < 0.7:
            # Single change
            if random.random() < 0.5:
                i = random.randint(0, N - 1)
                cur_a[i] = random.randint(0, N - 1)
            else:
                i = random.randint(0, N - 1)
                cur_b[i] = random.randint(0, N - 1)
        elif r < 0.85:
            # Swap two destinations in a
            i, j = random.sample(range(N), 2)
            cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
        elif r < 0.95:
            # Swap two destinations in b
            i, j = random.sample(range(N), 2)
            cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
        else:
            # Two independent changes
            i = random.randint(0, N - 1)
            cur_a[i] = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            cur_b[j] = random.randint(0, N - 1)
            
        new_score = compute_threshold_fast(cur_a, cur_b)
        if new_score == -1:
            new_score = -1000
            
        # Acceptance criterion
        accept = False
        if new_score > cur_score:
            accept = True
        else:
            delta = new_score - cur_score
            if delta < 0:
                # Metropolis: accept with probability exp(delta / T)
                # To avoid underflow, if delta/T < -100, prob is ~0
                if delta / T > -50:
                    prob = pow(2.718281828, delta / T)
                    if random.random() < prob:
                        accept = True
                # else prob is effectively 0
        
        if accept:
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = cur_a.copy()
                best_b = cur_b.copy()
        else:
            cur_a = prev_a
            cur_b = prev_b
            
        # Cool down
        T = max(T_min, T * alpha)
        
        # Occasional restart if stuck for too long?
        # For now, simple SA.
        
    return [best_a, best_b]
