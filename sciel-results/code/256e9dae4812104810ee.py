import time
import random
from collections import deque

N = 12

def calculate_threshold_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using bitmask representation.
    Much faster than frozenset approach.
    """
    start = (1 << N) - 1  # All states set
    
    # If start is already a single state (only possible if N=1, but N=12 here)
    # Check for single state: power of 2
    if (start & (start - 1)) == 0:
        return 0
    
    visited = {start: 0}
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        # Check if curr is a single state (power of 2)
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        # Apply letter 'a'
        next_a = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_a |= (1 << a_trans[idx])
            temp ^= lsb
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        # Apply letter 'b'
        next_b = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_b |= (1 << b_trans[idx])
            temp ^= lsb
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    # The evaluator kills after 90s. We use 85s as a hard stop to be safe.
    deadline = start_time + 85.0
    
    best_score = 0
    best_a = None
    best_b = None
    
    def try_automaton(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            return True
        return False

    # Strategy 1: Exhaustive search over structured families.
    # These are likely to contain high-threshold automata.
    
    # Family 1: a = cycle, b with controlled defects
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Single defect: b[i] = i for all i except b[k] = v
    # Note: If b is a permutation, it won't sync (unless trivial). 
    # In this family, if v != k, b is not a permutation (k is missing from image? No, k maps to v, v maps to v. 
    # Wait: b[k]=v. If v != k, then k is not in the image of b? 
    # Image: {0..N-1} \ {k} U {v}. Since v is already in {0..N-1}, the image size is N-1. 
    # So it is a non-permutation. Good.
    for k in range(N):
        for v in range(N):
            if v == k:
                continue
            if time.time() > deadline:
                break
            b = list(range(N))
            b[k] = v
            try_automaton(a_cycle, b)
    
    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle
            best_b = [0] * N
        return [best_a, best_b]
    
    # Family 2: Two defects in b (with a = cycle)
    # b[k1] = v1, b[k2] = v2, others identity.
    # We must ensure b is not a permutation.
    for k1 in range(N):
        for k2 in range(k1+1, N):
            for v1 in range(N):
                for v2 in range(N):
                    if time.time() > deadline:
                        break
                    # Skip if it results in a permutation
                    # A permutation requires all values distinct and covering 0..N-1.
                    # Since we change k1 and k2, we need to check if the set of values is a permutation.
                    # Simple check: create the list and check len(set)
                    b = list(range(N))
                    b[k1] = v1
                    b[k2] = v2
                    if len(set(b)) == N:
                        continue
                    try_automaton(a_cycle, b)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle
            best_b = [0] * N
        return [best_a, best_b]

    # Strategy 2: Random search with local improvement (Hill Climbing)
    # This allows escaping local minima in the threshold landscape.
    random.seed(42)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Generate random non-permutation b
        b = [random.randint(0, N-1) for _ in range(N)]
        # Ensure it's not a permutation
        if len(set(b)) == N:
            b[random.randint(0, N-1)] = b[random.randint(0, N-1)]
        
        a = a_cycle[:]
        
        if try_automaton(a, b):
            # Local search: try flipping one transition in b or a
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Modify b
                for i in range(N):
                    for new_val in range(N):
                        if new_val == b[i]:
                            continue
                        b_old = b[i]
                        b[i] = new_val
                        if len(set(b)) != N:  # Keep non-permutation
                            if try_automaton(a, b):
                                improved = True
                                break
                        b[i] = b_old
                    if improved:
                        break
                if improved:
                    continue
                    
                # Modify a
                for i in range(N):
                    for new_val in range(N):
                        if new_val == a[i]:
                            continue
                        a_old = a[i]
                        a[i] = new_val
                        if len(set(a)) != N:  # Keep non-permutation? 
                            # Actually, a doesn't strictly need to be non-permutation, 
                            # but if both are permutations, it won't sync.
                            # If a is perm and b is non-perm, it might sync.
                            # If a is non-perm and b is non-perm, it might sync.
                            # Let's just try it.
                            if try_automaton(a, b):
                                improved = True
                                break
                        a[i] = a_old
                    if improved:
                        break
    
    # Strategy 3: Try different 'a' structures with random 'b'
    if time.time() < deadline - 5.0:
        # Try a = identity
        a_id = list(range(N))
        for _ in range(2000):
            if time.time() > deadline:
                break
            b = [random.randint(0, N-1) for _ in range(N)]
            if len(set(b)) == N:
                b[random.randint(0, N-1)] = b[random.randint(0, N-1)]
            try_automaton(a_id, b)
        
        # Try a = reversal
        a_rev = [N - 1 - i for i in range(N)]
        for _ in range(2000):
            if time.time() > deadline:
                break
            b = [random.randint(0, N-1) for _ in range(N)]
            if len(set(b)) == N:
                b[random.randint(0, N-1)] = b[random.randint(0, N-1)]
            try_automaton(a_rev, b)

    # Fallback
    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N
    
    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above. 
# We keep the block markers for compatibility if the evaluator expects them, 
# but the logic is inside the function.
# EVOLVE-BLOCK-END
