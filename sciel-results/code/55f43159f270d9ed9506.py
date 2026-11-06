import time
import random
import sys

N = 12
FULL_MASK = (1 << N) - 1

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns 0 if not synchronizing.
    Optimized for N=12.
    """
    # Precompute transitions for all subsets
    # next_a[mask] = mask after applying 'a'
    # next_b[mask] = mask after applying 'b'
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    # DP for subset transitions
    # mask is a bitmask of states
    # next_trans[mask] = next_trans[mask without LSB] | (1 << trans[LSB_index])
    for mask in range(1, 1 << N):
        lb = mask & -mask
        i = (lb.bit_length() - 1)
        prev_mask = mask ^ lb
        
        next_a[mask] = next_a[prev_mask] | (1 << trans_a[i])
        next_b[mask] = next_b[prev_mask] | (1 << trans_b[i])
        
    # BFS from FULL_MASK to any singleton
    dist = [-1] * (1 << N)
    dist[FULL_MASK] = 0
    queue = [FULL_MASK]
    head = 0
    
    while head < len(queue):
        cur = queue[head]
        head += 1
        
        # Check if singleton
        if cur & (cur - 1) == 0:
            return dist[cur]
        
        d = dist[cur]
        
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = d + 1
            queue.append(nxt_a)
            
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = d + 1
            queue.append(nxt_b)
            
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # 1. Construct the Černý automaton
    # a: cycle 0->1->...->N-1->0
    # b: decrement, 0->0
    a_cerny = [(i + 1) % N for i in range(N)]
    b_cerny = [0] * N
    for i in range(1, N):
        b_cerny[i] = i - 1
        
    best_a = a_cerny[:]
    best_b = b_cerny[:]
    best_score = compute_threshold(best_a, best_b)
    
    # If we already have a very high score, we might not need to search much
    # Theoretical max for N=12 is likely 66.
    if best_score >= 66:
        return [best_a, best_b]
        
    # 2. Local Search to improve
    cur_a = best_a[:]
    cur_b = best_b[:]
    cur_score = best_score
    
    # Simple hill climbing / simulated annealing
    # We mutate one transition at a time
    temp = 5.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Random mutation
        state = random.randint(0, N-1)
        if random.random() < 0.5:
            old_dest = cur_a[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            cur_a[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            
            if new_score > cur_score:
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
                    temp = 5.0
            else:
                delta = new_score - cur_score
                if delta < 0:
                    p = pow(2.71828, delta / temp)
                    if random.random() < p:
                        cur_score = new_score
                    else:
                        cur_a[state] = old_dest
                else:
                    cur_score = new_score
        else:
            old_dest = cur_b[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            cur_b[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            
            if new_score > cur_score:
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
                    temp = 5.0
            else:
                delta = new_score - cur_score
                if delta < 0:
                    p = pow(2.71828, delta / temp)
                    if random.random() < p:
                        cur_score = new_score
                    else:
                        cur_b[state] = old_dest
                else:
                    cur_score = new_score
                    
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
            
    return [best_a, best_b]
