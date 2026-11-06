import time
import random

N = 12
FULL_MASK = (1 << N) - 1

def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns 0 if not synchronizing.
    Uses DP to compute subset transitions in O(2^N) time.
    """
    # Precompute transition of subsets using DP
    # next_a[mask] is the resulting mask after applying 'a'
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    # DP: next_trans[mask] = next_trans[mask without LSB] | (1 << trans[LSB_index])
    for mask in range(1, 1 << N):
        lb = mask & -mask
        i = (lb.bit_length() - 1)
        prev_mask = mask ^ lb
        
        next_a[mask] = next_a[prev_mask] | (1 << trans_a[i])
        next_b[mask] = next_b[prev_mask] | (1 << trans_b[i])
        
    # BFS from FULL_MASK
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
        
        nxt_a = next_a[cur]
        nxt_b = next_b[cur]
        
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
            
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
            
    return 0 # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    candidates = []
    
    # Candidate 1: a=cycle, b maps i to 0 if i%2==0, i to i+1 if i%2==1?
    # Just run local search from a few heuristic starts.
    
    starts = []
    # Start 1: Cycle + simple reset
    a1 = [(i + 1) % N for i in range(N)]
    b1 = [0] * N
    starts.append((a1, b1))
    
    # Start 2: Cycle + parity reset
    a2 = [(i + 1) % N for i in range(N)]
    b2 = [i % 2 for i in range(N)]
    starts.append((a2, b2))
    
    # Start 3: Cycle + random reset
    a3 = [(i + 1) % N for i in range(N)]
    b3 = [random.randint(0, N-1) for _ in range(N)]
    starts.append((a3, b3))
    
    # Start 4: Cycle + fold
    a4 = [(i + 1) % N for i in range(N)]
    b4 = [i // 2 if i < N-1 else N-1 for i in range(N)]
    starts.append((a4, b4))

    best_score = 0
    best_a = None
    best_b = None
    
    for a_start, b_start in starts:
        score = compute_threshold(a_start, b_start)
        if score > best_score:
            best_score = score
            best_a = a_start[:]
            best_b = b_start[:]
            
        # Local Search from this start
        cur_a = a_start[:]
        cur_b = b_start[:]
        cur_score = score
        
        iters = 0
        while time.time() < deadline:
            iters += 1
            # Mutation
            # 1. Change single transition
            # 2. Swap two transitions in same letter
            # 3. Copy transition from one state to another
            
            move_type = random.random()
            
            if move_type < 0.7:
                # Single change
                state = random.randint(0, N-1)
                if random.random() < 0.5:
                    old_dest = cur_a[state]
                    new_dest = random.randint(0, N-1)
                    if new_dest == old_dest: continue
                    cur_a[state] = new_dest
                    new_score = compute_threshold(cur_a, cur_b)
                    if new_score <= cur_score:
                        cur_a[state] = old_dest
                    else:
                        cur_score = new_score
                        if cur_score > best_score:
                            best_score = cur_score
                            best_a = cur_a[:]
                            best_b = cur_b[:]
                else:
                    old_dest = cur_b[state]
                    new_dest = random.randint(0, N-1)
                    if new_dest == old_dest: continue
                    cur_b[state] = new_dest
                    new_score = compute_threshold(cur_a, cur_b)
                    if new_score <= cur_score:
                        cur_b[state] = old_dest
                    else:
                        cur_score = new_score
                        if cur_score > best_score:
                            best_score = cur_score
                            best_a = cur_a[:]
                            best_b = cur_b[:]
            elif move_type < 0.9:
                # Swap
                if random.random() < 0.5:
                    i, j = random.sample(range(N), 2)
                    cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
                    new_score = compute_threshold(cur_a, cur_b)
                    if new_score <= cur_score:
                        cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
                    else:
                        cur_score = new_score
                        if cur_score > best_score:
                            best_score = cur_score
                            best_a = cur_a[:]
                            best_b = cur_b[:]
                else:
                    i, j = random.sample(range(N), 2)
                    cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
                    new_score = compute_threshold(cur_a, cur_b)
                    if new_score <= cur_score:
                        cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
                    else:
                        cur_score = new_score
                        if cur_score > best_score:
                            best_score = cur_score
                            best_a = cur_a[:]
                            best_b = cur_b[:]
            else:
                # Copy
                i, j = random.sample(range(N), 2)
                if random.random() < 0.5:
                    old_dest = cur_a[j]
                    cur_a[j] = cur_a[i]
                    new_score = compute_threshold(cur_a, cur_b)
                    if new_score <= cur_score:
                        cur_a[j] = old_dest
                    else:
                        cur_score = new_score
                        if cur_score > best_score:
                            best_score = cur_score
                            best_a = cur_a[:]
                            best_b = cur_b[:]
                else:
                    old_dest = cur_b[j]
                    cur_b[j] = cur_b[i]
                    new_score = compute_threshold(cur_a, cur_b)
                    if new_score <= cur_score:
                        cur_b[j] = old_dest
                    else:
                        cur_score = new_score
                        if cur_score > best_score:
                            best_score = cur_score
                            best_a = cur_a[:]
                            best_b = cur_b[:]

    # Ensure we return a synchronizing automaton
    if best_score == 0:
        # Fallback to seed
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return [best_a, best_b]
