import time
import random

N = 12
SIZE = 1 << N

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    """
    # Precompute transition for each subset for letter a and b
    size = SIZE
    trans_a_sub = [0] * size
    trans_b_sub = [0] * size
    
    for subset in range(size):
        if subset == 0:
            continue
        sa = 0
        sb = 0
        temp = subset
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= (1 << trans_a[i])
            sb |= (1 << trans_b[i])
            temp ^= lsb
        trans_a_sub[subset] = sa
        trans_b_sub[subset] = sb

    start = (1 << N) - 1
    if start == 1:
        return 0
        
    visited = [False] * size
    queue = [start]
    visited[start] = True
    dist = 0
    
    while queue:
        new_queue = []
        for subset in queue:
            # Check if singleton
            if subset & (subset - 1) == 0:
                return dist
            
            # Apply a
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                new_queue.append(next_a)
            
            # Apply b
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                new_queue.append(next_b)
        
        queue = new_queue
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 55 # Safety margin (evaluator limit 90s, but keep buffer)
    
    # Strategy:
    # 1. Initialize with a "de Bruijn-like" or specific high-threshold structure if known,
    #    otherwise use a robust random start that is likely synchronizing.
    #    The "Petr" automaton structure (cycle with a twist) is a good candidate.
    #    Let's try a specific known good family:
    #    a: i -> i+1 mod N
    #    b: 0->1, i->i+1 for i>0? No.
    
    # Let's use a mix of random restarts and hill climbing.
    # To improve the initial score, we will run a short random search to find a good baseline.
    
    best_a = None
    best_b = None
    best_len = -1
    
    # Phase 1: Find a good starting point (short time budget)
    # We try a few specific patterns and random ones.
    start_time = time.time()
    phase1_deadline = start_time + 5.0
    
    # Pattern 1: Cycle + specific perturbation
    # a = [1, 2, ..., N-1, 0]
    # b = [0, 0, ..., 0, 0] is length 1.
    # Let's try b = [0, 1, 2, ..., N-2, 0]?
    
    candidates = []
    
    # Generate some structured candidates
    base_a = [(i + 1) % N for i in range(N)]
    
    # Candidate: a is cycle, b is "almost" cycle but maps last to 0
    b1 = [0] * N
    for i in range(N):
        b1[i] = (i + 1) % N
    b1[N-1] = 0
    candidates.append((base_a[:], b1[:]))
    
    # Candidate: a is cycle, b maps 0->0, i->i-1?
    b2 = [0] * N
    for i in range(1, N):
        b2[i] = i - 1
    candidates.append((base_a[:], b2[:]))
    
    # Random candidates
    for _ in range(20):
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        candidates.append((a, b))
        
    for a, b in candidates:
        if time.time() > phase1_deadline:
            break
        l = compute_reset_length(a, b)
        if l > best_len:
            best_len = l
            best_a = a[:]
            best_b = b[:]
            
    if best_a is None:
        # Fallback: a known synchronizing automaton
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_len = compute_reset_length(best_a, best_b)
        
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    # Phase 2: Local Search (Hill Climbing with Random Restarts/Kicks)
    no_improve_count = 0
    kick_threshold = 200 # Lower threshold to encourage exploration
    
    while time.time() < deadline:
        # Choose a transition to mutate
        if random.random() < 0.5:
            i = random.randint(0, N-1)
            old_val = current_a[i]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            current_a[i] = new_val
            new_len = compute_reset_length(current_a, current_b)
            if new_len == -1: # Not synchronizing
                current_a[i] = old_val
                no_improve_count += 1
                continue
            if new_len > current_len:
                current_len = new_len
                if new_len > best_len:
                    best_len = new_len
                    best_a = current_a[:]
                    best_b = current_b[:]
                    no_improve_count = 0
            else:
                current_a[i] = old_val
                no_improve_count += 1
        else:
            i = random.randint(0, N-1)
            old_val = current_b[i]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            current_b[i] = new_val
            new_len = compute_reset_length(current_a, current_b)
            if new_len == -1:
                current_b[i] = old_val
                no_improve_count += 1
                continue
            if new_len > current_len:
                current_len = new_len
                if new_len > best_len:
                    best_len = new_len
                    best_a = current_a[:]
                    best_b = current_b[:]
                    no_improve_count = 0
            else:
                current_b[i] = old_val
                no_improve_count += 1
                
        # Kick if stuck
        if no_improve_count > kick_threshold:
            # Randomize current state partially to escape local maxima
            # Instead of full random, perturb a few random transitions
            for _ in range(3):
                if random.random() < 0.5:
                    current_a[random.randint(0, N-1)] = random.randint(0, N-1)
                else:
                    current_b[random.randint(0, N-1)] = random.randint(0, N-1)
            
            # Check if it's still valid (synchronizing)
            new_len = compute_reset_length(current_a, current_b)
            if new_len == -1:
                # Revert to best
                current_a = best_a[:]
                current_b = best_b[:]
                current_len = best_len
            else:
                current_len = new_len
            no_improve_count = 0
            
    return [best_a, best_b]
