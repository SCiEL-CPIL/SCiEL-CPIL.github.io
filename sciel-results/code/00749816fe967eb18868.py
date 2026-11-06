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
    Optimized for speed: precomputes subset transitions.
    """
    # Precompute transitions for all subsets
    trans_a_sub = [0] * SIZE
    trans_b_sub = [0] * SIZE
    
    for subset in range(1, SIZE):
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

    # Strategy:
    # 1. Systematic enumeration of the "Pinwheel" family.
    #    Structure:
    #    a: (i+1) % N
    #    b: i -> i-1 for all i, except a set of "breaks" where b(i) -> 0 (or similar sink)
    #    
    #    The "breaks" determine the reset length.
    #    We enumerate all subsets of states to be "broken" in b.
    #    There are 2^N = 4096 subsets. This is very fast.
    
    # Base a: Cycle
    base_a = [(i + 1) % N for i in range(N)]
    
    # Base b: Reverse cycle (i-1)%N, but we will modify specific indices to point to 0.
    # This creates a "star" sink at 0 for those indices, which helps synchronization.
    
    # We also consider variations where the sink is not 0, or where a is not a simple cycle.
    # But the cycle + broken reverse cycle is the known high-performing family for small N.
    
    # To be thorough, we'll iterate over all masks of "breaks" in b.
    # If bit k is set in mask, b[k] = 0. Otherwise b[k] = (k-1)%N.
    
    # We also try a few variations of 'a' to see if they help, but usually a=cycle is optimal for this family.
    
    for mask in range(1, 1 << N): # Mask 0 is pure reverse cycle (not synchronizing usually or low threshold)
        b = [(i - 1) % N for i in range(N)]
        for i in range(N):
            if mask & (1 << i):
                b[i] = 0
        
        # Check time
        if time.time() > deadline - 5:
            break
            
        l = compute_reset_length(base_a, b)
        if l > best_len:
            best_len = l
            best_a = base_a[:]
            best_b = b[:]

    # If we still have time, try other structural families.
    # Family 2: "Rank 2" constructions or different cycle lengths.
    # Let's try a few random restarts with local search if time permits, 
    # but only starting from the best found so far.
    
    if time.time() < deadline - 10 and best_a is not None:
        # Local refinement of the best found
        # We will try flipping individual bits in the mask of the best b
        # or changing the sink target.
        
        current_a = best_a[:]
        current_b = best_b[:]
        current_len = best_len
        
        # Simple hill climbing on the mask of breaks
        # Reconstruct mask from current_b?
        # It's easier to just try random mutations on current_b
        
        no_improve = 0
        max_no_improve = 20
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            a_child = current_a[:]
            b_child = current_b[:]
            
            # Random mutation
            if random.random() < 0.5:
                # Change one b transition
                i = random.randint(0, N-1)
                b_child[i] = random.randint(0, N-1)
            else:
                # Change one a transition
                i = random.randint(0, N-1)
                a_child[i] = random.randint(0, N-1)
                
            child_len = compute_reset_length(a_child, b_child)
            
            if child_len > current_len:
                current_a = a_child
                current_b = b_child
                current_len = child_len
                no_improve = 0
                if child_len > best_len:
                    best_len = child_len
                    best_a = a_child[:]
                    best_b = b_child[:]
            else:
                no_improve += 1

    # Fallback if nothing found (should not happen)
    if best_a is None:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        for i in range(1, N):
            best_b[i] = i - 1

    return [best_a, best_b]
