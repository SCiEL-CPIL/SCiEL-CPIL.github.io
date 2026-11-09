import time
import random
import math

N = 12
MAX_TIME = 80.0  # Safety margin before 90s kill

def compute_threshold_fast(a_map, b_map):
    """
    Computes the shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    size = 1 << N
    trans_a = [0] * size
    trans_b = [0] * size
    
    # Precompute transitions for all masks
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
    random.seed(42)
    
    # Start with Pawlik automaton
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] + [i - 1 for i in range(1, N)]
    
    best_a = a_map.copy()
    best_b = b_map.copy()
    best_score = compute_threshold_fast(best_a, best_b)
    
    cur_a = a_map.copy()
    cur_b = b_map.copy()
    cur_score = best_score
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.1
    alpha = 0.9995
    
    # Additional: try a few different initializations
    initializations = [
        ([(i + 1) % N for i in range(N)], [0] + [i - 1 for i in range(1, N)]),  # Pawlik
        ([random.randint(0, N-1) for _ in range(N)], [random.randint(0, N-1) for _ in range(N)]),
        ([random.randint(0, N-1) for _ in range(N)], [random.randint(0, N-1) for _ in range(N)]),
    ]
    
    # Try each initialization
    for init_a, init_b in initializations:
        if time.time() - start_time > MAX_TIME - 5:
            break
            
        cur_a = init_a.copy()
        cur_b = init_b.copy()
        cur_score = compute_threshold_fast(cur_a, cur_b)
        
        if cur_score == -1:
            cur_score = -1000
        else:
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a.copy()
                best_b = cur_b.copy()
        
        # Run SA for a portion of the time
        sa_start = time.time()
        sa_duration = min(20.0, MAX_TIME - (sa_start - start_time) - 5)
        
        while time.time() - sa_start < sa_duration:
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
                    try:
                        prob = math.exp(delta / T)
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
            
        # Reset temperature for next initialization
        T = 10.0
    
    return [best_a, best_b]
