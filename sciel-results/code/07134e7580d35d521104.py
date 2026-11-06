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
    
    best_a = None
    best_b = None
    best_len = -1
    
    def evaluate(a, b):
        nonlocal best_a, best_b, best_len
        l = compute_reset_length(a, b)
        if l > best_len:
            best_len = l
            best_a = a[:]
            best_b = b[:]

    # Phase 1: Systematic Enumeration of Known High-Threshold Families
    # These structures are known to approach the Černý bound N^2 - N + 4.
    # For N=12, the bound is 148. Known constructions often reach ~100-120.
    
    candidates = []
    
    # 1. Pinwheel Family (often optimal or near-optimal for binary alphabets)
    # Structure: a is a cycle (0->1->...->N-1->0).
    # b maps most states to i+1, but "breaks" the cycle in a specific way.
    # Standard Pinwheel: b(i) = i+1 for i < N-1, b(N-1) = 0 is just a cycle (not synchronizing).
    # We need a specific perturbation.
    
    # Variant A: "Short Pinwheel"
    # a: i -> i+1 mod N
    # b: i -> i+1 mod N for i != N-1, b(N-1) -> 0
    # This is just two cycles, not synchronizing.
    
    # Let's use the specific construction from "On the Černý conjecture" literature.
    # A common high-threshold family is defined by:
    # a: i -> (i+1) % N
    # b: i -> (i+1) % N for i < N-1, b(N-1) -> 0
    # Wait, that's not it.
    
    # Let's generate a diverse set of structured candidates based on cycle perturbations.
    
    # Family 1: Cycle with a single "reset" edge
    a_cycle = [(i + 1) % N for i in range(N)]
    for k in range(N):
        for target in range(N):
            b = a_cycle[:]
            b[k] = target
            candidates.append((a_cycle[:], b))

    # Family 2: Two cycles interacting
    # a: 0->1->...->N/2-1->0 and N/2->N/2+1->...->N-1->N/2
    # This doesn't synchronize easily.
    
    # Family 3: "De Bruijn" style or random-like but structured
    # Let's add a few high-quality random restarts that are checked quickly.
    # But first, let's try the specific "Rank-2" constructions.
    
    # Rank-2 Construction (from Pinwheel paper):
    # States 0..N-1.
    # a: i -> i+1 mod N
    # b: 0 -> 0, i -> i+1 for i>0? No.
    
    # Let's use a heuristic generator for "almost cycles" which tend to have long reset words.
    # An "almost cycle" has N-1 edges forming a cycle and 1 edge pointing into it.
    
    # Generate all "almost cycles" for 'a' and test with a fixed 'b' that breaks the cycle.
    # Actually, it's better to vary both.
    
    # Given the time limit, we can't test all N^4 combinations of single perturbations.
    # But we can test specific known good ones.
    
    # Known Good: 
    # a: 0->1, 1->2, ..., N-2->N-1, N-1->0 (Cycle)
    # b: 0->0, 1->0, 2->1, ..., N-1->N-2 (Shifts down, maps 0 to 0)
    # This is a known structure. Let's evaluate it.
    b_shift = [0] * N
    for i in range(1, N):
        b_shift[i] = i - 1
    candidates.append((a_cycle[:], b_shift))
    
    # Another known:
    # a: Cycle
    # b: 0->0, i->i (Identity) -> Not synchronizing.
    
    # Let's add random candidates to the pool, but limit the count to keep Phase 1 fast.
    random.seed(42)
    for _ in range(50):
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        candidates.append((a, b))

    # Evaluate all structured candidates
    for a, b in candidates:
        if time.time() > deadline - 10: # Keep 10s for Phase 2 if needed, or just break
            break
        evaluate(a, b)
        
    if best_a is None:
        # Fallback
        best_a = a_cycle[:]
        best_b = b_shift[:]
        evaluate(best_a, best_b)
        
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    # Phase 2: Local Search (Hill Climbing with Random Restarts/Kicks)
    # Use remaining time to try to improve.
    no_improve_count = 0
    kick_threshold = 50
    
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
            for _ in range(3):
                if random.random() < 0.5:
                    current_a[random.randint(0, N-1)] = random.randint(0, N-1)
                else:
                    current_b[random.randint(0, N-1)] = random.randint(0, N-1)
            
            new_len = compute_reset_length(current_a, current_b)
            if new_len == -1:
                current_a = best_a[:]
                current_b = best_b[:]
                current_len = best_len
            else:
                current_len = new_len
            no_improve_count = 0
            
    return [best_a, best_b]
