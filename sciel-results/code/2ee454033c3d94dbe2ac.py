import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def calculate_threshold_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using bitmask representation.
    Precomputes transition images for single states to speed up subset transitions.
    """
    # Precompute: for each state, where it goes under a and b
    # img_a[s] = 1 << a_trans[s]
    # img_b[s] = 1 << b_trans[s]
    img_a = [1 << a_trans[i] for i in range(N)]
    img_b = [1 << b_trans[i] for i in range(N)]
    
    start = FULL_MASK
    if (start & (start - 1)) == 0:
        return 0
    
    visited = {start: 0}
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        if (curr & (curr - 1)) == 0:  # Single state
            return curr_dist
        
        # Apply letter 'a'
        next_a = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_a |= img_a[idx]
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
            next_b |= img_b[idx]
            temp ^= lsb
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # 3s safety margin
    
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
    
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Strategy 1: Structured search over defect patterns
    # Family 1: a = cycle, b = identity with k defects
    # One defect
    for k in range(N):
        for v in range(N):
            if v == k:
                continue
            b = list(range(N))
            b[k] = v
            try_automaton(a_cycle, b)
    
    # Two defects
    for k1 in range(N):
        for k2 in range(k1 + 1, N):
            for v1 in range(N):
                for v2 in range(N):
                    if v1 == k1 and v2 == k2:
                        continue
                    if time.time() > deadline - 30:
                        break
                    b = list(range(N))
                    b[k1] = v1
                    b[k2] = v2
                    if len(set(b)) == N:
                        continue
                    try_automaton(a_cycle, b)
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break
    
    # Strategy 2: Random search with local improvement
    random.seed(12345)
    
    # Try multiple 'a' structures
    a_structures = [
        a_cycle,
        list(range(N)),  # identity
        [N - 1 - i for i in range(N)],  # reversal
        [(i * 5) % N for i in range(N)],  # multiplier (gcd(5,12)=1, so permutation)
    ]
    
    for a_base in a_structures:
        if time.time() > deadline - 10:
            break
        
        for _ in range(3000):
            if time.time() > deadline - 5:
                break
            
            # Generate random non-permutation b
            b = [random.randint(0, N - 1) for _ in range(N)]
            if len(set(b)) == N:
                # Force a collision
                j = random.randint(0, N - 1)
                b[j] = b[random.randint(0, N - 1)]
            
            if try_automaton(a_base, b):
                # Local search: try changing one transition in b
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(N):
                        if time.time() > deadline - 5:
                            break
                        old_val = b[i]
                        for new_val in range(N):
                            if new_val == old_val:
                                continue
                            b[i] = new_val
                            if len(set(b)) != N:
                                if try_automaton(a_base, b):
                                    improved = True
                                    break
                        b[i] = old_val
                    if time.time() > deadline - 5:
                        break
    
    # Strategy 3: More structured random search - random a, random b
    if time.time() < deadline - 5:
        for _ in range(2000):
            if time.time() > deadline - 3:
                break
            
            # Random a (non-permutation preferred for sync)
            a = [random.randint(0, N - 1) for _ in range(N)]
            if len(set(a)) == N:
                j = random.randint(0, N - 1)
                a[j] = a[random.randint(0, N - 1)]
            
            # Random b (non-permutation)
            b = [random.randint(0, N - 1) for _ in range(N)]
            if len(set(b)) == N:
                j = random.randint(0, N - 1)
                b[j] = b[random.randint(0, N - 1)]
            
            try_automaton(a, b)
    
    # Fallback
    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N
    
    return [best_a, best_b]
