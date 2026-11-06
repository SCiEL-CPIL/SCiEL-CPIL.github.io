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
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    # Precompute transitions for all subsets
    # Using local variables for speed
    ta = trans_a
    tb = trans_b
    na = next_a
    nb = next_b
    
    for mask in range(1, 1 << N):
        lb = mask & -mask
        i = lb.bit_length() - 1
        prev_mask = mask ^ lb
        
        na[mask] = na[prev_mask] | (1 << ta[i])
        nb[mask] = nb[prev_mask] | (1 << tb[i])
        
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
        
        nxt_a = na[cur]
        nxt_b = nb[cur]
        
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
            
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
            
    return 0 # Not synchronizing

def generate_heuristic_seed():
    """
    Generates a diverse heuristic seed for the automaton.
    """
    strategy = random.randint(0, 4)
    
    if strategy == 0:
        # Classic Černý: Cycle + Shift-Down
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        for i in range(1, N):
            b[i] = i - 1
        return a, b
        
    elif strategy == 1:
        # Cycle + Random Reset
        a = [(i + 1) % N for i in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        return a, b
        
    elif strategy == 2:
        # Pinwheel: a is cycle, b maps i to i-1 mod N (but this is just a cycle, need to break it)
        # Actually, standard pinwheel often has a cycle and a map that folds it.
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        # Randomly assign some states to 0 to create a "sink" behavior
        for i in range(N):
            if random.random() < 0.3:
                b[i] = 0
            else:
                b[i] = (i - 1) % N
        return a, b
        
    elif strategy == 3:
        # Random Permutation for A, Random for B
        a = list(range(N))
        random.shuffle(a)
        b = [random.randint(0, N-1) for _ in range(N)]
        return a, b
        
    else:
        # "Fold" structure: a is cycle, b maps i to i//2
        a = [(i + 1) % N for i in range(N)]
        b = [i // 2 for i in range(N)]
        return a, b

def construct():
    start_time = time.time()
    deadline = start_time + 58.0 # 2s margin for safety
    
    best_score = 0
    best_a = None
    best_b = None
    
    # Initial Evaluation of Heuristic Seeds
    for _ in range(20):
        a_seed, b_seed = generate_heuristic_seed()
        score = compute_threshold(a_seed, b_seed)
        if score > best_score:
            best_score = score
            best_a = a_seed[:]
            best_b = b_seed[:]
            
    if best_a is None:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        for i in range(1, N):
            best_b[i] = i - 1
        best_score = compute_threshold(best_a, best_b)
        
    cur_a = best_a[:]
    cur_b = best_b[:]
    cur_score = best_score
    
    # Local Search with Restarts
    # We perform hill climbing. If we get stuck, we restart from a new seed or mutated best.
    stuck_counter = 0
    max_stuck = 500 # Number of failed moves before restart
    
    while time.time() < deadline:
        # Check time more frequently in the inner loop if stuck
        if stuck_counter >= max_stuck:
            # Restart strategy
            if random.random() < 0.5:
                # New heuristic seed
                a_seed, b_seed = generate_heuristic_seed()
                cur_a = a_seed[:]
                cur_b = b_seed[:]
                cur_score = compute_threshold(cur_a, cur_b)
            else:
                # Heavily mutate current best
                cur_a = best_a[:]
                cur_b = best_b[:]
                for _ in range(5):
                    state = random.randint(0, N-1)
                    if random.random() < 0.5:
                        cur_a[state] = random.randint(0, N-1)
                    else:
                        cur_b[state] = random.randint(0, N-1)
                cur_score = compute_threshold(cur_a, cur_b)
            
            stuck_counter = 0
            continue
        
        # Mutation
        # 1. Single transition change
        state = random.randint(0, N-1)
        if random.random() < 0.5:
            old_dest = cur_a[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            cur_a[state] = new_dest
        else:
            old_dest = cur_b[state]
            new_dest = random.randint(0, N-1)
            if new_dest == old_dest:
                continue
            cur_b[state] = new_dest
            
        new_score = compute_threshold(cur_a, cur_b)
        
        if new_score > cur_score:
            cur_score = new_score
            stuck_counter = 0
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a[:]
                best_b = cur_b[:]
        else:
            # Reject: revert
            if random.random() < 0.5:
                cur_a[state] = old_dest
            else:
                cur_b[state] = old_dest
            stuck_counter += 1
            
            # Occasional "kick" to escape local minima even if not fully stuck
            if stuck_counter % 100 == 0:
                # Swap two transitions
                i, j = random.sample(range(N), 2)
                if random.random() < 0.5:
                    cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
                else:
                    cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                    stuck_counter = 0
                else:
                    # Revert swap
                    if random.random() < 0.5:
                        cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
                    else:
                        cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
    
    # Final Check: Ensure we return a valid synchronizing automaton
    if best_score == 0:
        # Fallback: Return a known synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return [best_a, best_b]
