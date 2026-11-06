import time
import random
import copy

N = 12
SIZE = 1 << N

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    """
    # Precompute transitions for all subsets
    trans_a_sub = [0] * SIZE
    trans_b_sub = [0] * SIZE
    
    # Optimization: Precompute for single states first, then combine? 
    # Actually, the bitwise OR property means we can just iterate.
    # But 4096 subsets * 12 states = ~50k ops. Very fast.
    
    for subset in range(SIZE):
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
        
    visited = [False] * SIZE
    queue = [start]
    visited[start] = True
    dist = 0
    
    while queue:
        new_queue = []
        for subset in queue:
            if subset & (subset - 1) == 0:
                return dist
            
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                new_queue.append(next_a)
            
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                new_queue.append(next_b)
        
        queue = new_queue
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 55  # Safety margin for the 60s limit
    
    # Strategy:
    # 1. Enumerate specific structural families known to have high thresholds.
    # 2. Use a targeted local search (hill climbing) starting from these structures.
    # 3. The "Pinwheel" family is the primary candidate.
    #    Structure:
    #    a: i -> (i+1) % N
    #    b: i -> i (for i in S), i -> i-1 (for others) OR similar variations.
    #    
    #    Known best for N=12 is likely in the range 60-100+.
    #    The seed program scored 112. Let's try to beat it.
    
    best_a = None
    best_b = None
    best_len = -1
    
    def update_best(a, b):
        nonlocal best_a, best_b, best_len
        l = compute_reset_length(a, b)
        if l > best_len:
            best_len = l
            best_a = a[:]
            best_b = b[:]
        return l

    # Family 1: Standard Pinwheel variants
    # a: cycle
    # b: "almost identity" or "shift down with breaks"
    
    # Let's generate a diverse set of initial candidates
    candidates = []
    
    # Candidate 1: The "classic" pinwheel
    a1 = [(i + 1) % N for i in range(N)]
    b1 = [0] * N
    for i in range(1, N):
        b1[i] = i - 1
    candidates.append((a1[:], b1[:]))
    
    # Candidate 2: Variation where b maps 0->N-1 (full cycle) but one break
    a2 = [(i + 1) % N for i in range(N)]
    b2 = [(i - 1) % N for i in range(N)]
    b2[0] = 0 # Break at 0
    candidates.append((a2[:], b2[:]))
    
    # Candidate 3: Another break position
    a3 = [(i + 1) % N for i in range(N)]
    b3 = [(i - 1) % N for i in range(N)]
    b3[N-1] = 0 # Break at N-1
    candidates.append((a3[:], b3[:]))
    
    # Candidate 4: Double break
    a4 = [(i + 1) % N for i in range(N)]
    b4 = [(i - 1) % N for i in range(N)]
    b4[0] = 0
    b4[5] = 0
    candidates.append((a4[:], b4[:]))
    
    # Candidate 5: "Star" like b
    a5 = [(i + 1) % N for i in range(N)]
    b5 = [0] * N # All to 0
    candidates.append((a5[:], b5[:])) # This will be fast (threshold 1 or 2) but valid
    
    # Candidate 6: "Comet" like b
    a6 = [(i + 1) % N for i in range(N)]
    b6 = list(range(N)) # Identity
    b6[0] = 0
    b6[1] = 0
    candidates.append((a6[:], b6[:]))
    
    # Randomized Pinwheels
    for _ in range(50):
        a_r = [(i + 1) % N for i in range(N)]
        b_r = [(i - 1) % N for i in range(N)]
        # Random breaks
        num_breaks = random.randint(1, 3)
        for _ in range(num_breaks):
            idx = random.randint(0, N-1)
            # Map to 0 or a low number to ensure synchronizability
            b_r[idx] = random.randint(0, 2)
        candidates.append((a_r[:], b_r[:]))

    # Evaluate initial candidates
    for a, b in candidates:
        update_best(a, b)
        
    if time.time() > deadline - 5:
        if best_a is None:
            best_a = candidates[0][0]
            best_b = candidates[0][1]
        return [best_a, best_b]

    # Local Search (Hill Climbing)
    # We will pick the best candidate found so far and try to improve it.
    # Also, we will keep a small population of the top K candidates to avoid local optima.
    
    # Top K population
    pop = []
    for a, b in candidates:
        l = compute_reset_length(a, b)
        pop.append((l, a[:], b[:]))
    
    pop.sort(key=lambda x: x[0], reverse=True)
    pop = pop[:20] # Keep top 20
    
    # Hill Climbing Loop
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        # Pick a random parent from the top few
        idx = random.randint(0, min(5, len(pop)-1))
        _, a, b = pop[idx]
        
        # Generate a child via mutation
        # Mutation strategies:
        # 1. Change one transition in a
        # 2. Change one transition in b
        # 3. Swap two transitions in a or b
        
        a_child = a[:]
        b_child = b[:]
        
        mut_type = random.randint(0, 2)
        
        if mut_type == 0:
            # Single bit flip in a
            i = random.randint(0, N-1)
            a_child[i] = random.randint(0, N-1)
        elif mut_type == 1:
            # Single bit flip in b
            i = random.randint(0, N-1)
            b_child[i] = random.randint(0, N-1)
        else:
            # Swap in b (maintains permutation-like structure if it was one)
            i1, i2 = random.sample(range(N), 2)
            b_child[i1], b_child[i2] = b_child[i2], b_child[i1]
            
        # Evaluate child
        child_len = compute_reset_length(a_child, b_child)
        
        if child_len == -1:
            # Not synchronizing, discard or repair? 
            # For speed, discard.
            no_improve_count += 1
            continue
            
        if child_len > best_len:
            best_len = child_len
            best_a = a_child[:]
            best_b = b_child[:]
            no_improve_count = 0
            # Add to population
            pop.append((child_len, a_child[:], b_child[:]))
            pop.sort(key=lambda x: x[0], reverse=True)
            pop = pop[:20]
        elif child_len > pop[-1][0]:
            # Better than worst in pop, replace
            pop[-1] = (child_len, a_child[:], b_child[:])
            pop.sort(key=lambda x: x[0], reverse=True)
            no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Occasionally, inject a fresh random candidate to escape local optima
        if random.random() < 0.05:
            a_r = [(i + 1) % N for i in range(N)]
            b_r = [(i - 1) % N for i in range(N)]
            for _ in range(random.randint(1, 3)):
                idx_b = random.randint(0, N-1)
                b_r[idx_b] = random.randint(0, 2)
            r_len = compute_reset_length(a_r, b_r)
            if r_len != -1:
                pop.append((r_len, a_r[:], b_r[:]))
                pop.sort(key=lambda x: x[0], reverse=True)
                pop = pop[:20]
                if r_len > best_len:
                    best_len = r_len
                    best_a = a_r[:]
                    best_b = b_r[:]
                    no_improve_count = 0

    if best_a is None:
        # Fallback
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        for i in range(1, N):
            best_b[i] = i - 1

    return [best_a, best_b]
