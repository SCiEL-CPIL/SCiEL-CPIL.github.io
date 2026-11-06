import time
import math

def construct():
    start_time = time.time()
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
        
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    # states[i] is list of (gap, length)
    # parents[i] is list of (prev_index, prev_state_index)
    states = [[] for _ in range(N)]
    parents = [[] for _ in range(N)]
    
    for i in range(N):
        states[i].append((0, 1))
        parents[i].append((-1, -1))
        
    for i in range(N):
        if time.time() - start_time > 55:
            break
        # Sort states[i] by gap (already sorted from previous steps? No, we sort after updates)
        # We assume states[i] is maintained sorted by gap and length is increasing.
        for j in range(i + 1, N):
            G = primes[j] - primes[i]
            # Find valid states in i: gap <= G
            # states[i] is sorted by gap.
            # Find rightmost index in states[i] with gap <= G
            # Binary search
            lo, hi = 0, len(states[i]) - 1
            last_valid_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if states[i][mid][0] <= G:
                    last_valid_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if last_valid_idx == -1:
                continue
                
            for k in range(last_valid_idx + 1):
                G_last, L_last = states[i][k]
                new_L = L_last + 1
                new_G = G
                
                # Check dominance in states[j]
                # states[j] is sorted by gap, length increasing.
                # Find pos in states[j] where gap <= new_G
                lo, hi = 0, len(states[j]) - 1
                pos = -1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if states[j][mid][0] <= new_G:
                        pos = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                        
                # Check if dominated by states[j][pos]
                if pos != -1 and states[j][pos][1] >= new_L:
                    continue
                    
                # Remove dominated states from states[j]
                # Dominated if new_G <= G_e and new_L >= L_e
                # Find start_remove: first index with gap >= new_G
                lo, hi = 0, len(states[j]) - 1
                start_remove = len(states[j])
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if states[j][mid][0] >= new_G:
                        start_remove = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                        
                # Find end_remove: first index >= start_remove with length > new_L
                # Since length increases with gap in the frontier, we can binary search length.
                # But we only care about indices >= start_remove.
                # The lengths in states[j][start_remove:] are increasing.
                lo, hi = start_remove, len(states[j]) - 1
                end_remove = len(states[j])
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if states[j][mid][1] > new_L:
                        end_remove = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                        
                # states[j][start_remove:end_remove] are dominated
                # Construct new list
                new_states_j = states[j][:start_remove] + [(new_G, new_L)] + states[j][end_remove:]
                new_parents_j = parents[j][:start_remove] + [(i, k)] + parents[j][end_remove:]
                
                states[j] = new_states_j
                parents[j] = new_parents_j
                
    # Find best
    max_len = 0
    best_end = 0
    best_state_idx = 0
    for i in range(N):
        for k, (g, l) in enumerate(states[i]):
            if l > max_len:
                max_len = l
                best_end = i
                best_state_idx = k
                
    path = []
    cur_i = best_end
    cur_k = best_state_idx
    while cur_i != -1:
        path.append(cur_i)
        prev_i, prev_k = parents[cur_i][cur_k]
        cur_i = prev_i
        cur_k = prev_k
        
    path.reverse()
    return [primes[idx] for idx in path]
