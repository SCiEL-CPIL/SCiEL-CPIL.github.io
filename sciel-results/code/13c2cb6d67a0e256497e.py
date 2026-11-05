import time
import random
from collections import deque

N = 12

def calculate_threshold_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using bitmask representation.
    """
    start = (1 << N) - 1  # All states set
    if start == 0 or (start & (start - 1)) == 0:
        return 0
    
    # Precompute transition lookup for each subset size isn't feasible,
    # but we can speed up the inner loop by using a different approach.
    
    visited = {start: 0}
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        if (curr & (curr - 1)) == 0:  # Power of 2 = single state
            return curr_dist
        
        # Apply letter 'a': use bit manipulation to compute image
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
    deadline = start_time + 78.0  # Safety margin for evaluation overhead
    
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
    
    # Strategy 1: Try known high-threshold families
    # Family 1: a = cycle, b with controlled defects
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Single defect: b[i] = i for all i except b[k] = v
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
    
    # Family 2: Two defects
    for k1 in range(N):
        for k2 in range(k1+1, N):
            for v1 in range(N):
                for v2 in range(N):
                    if time.time() > deadline:
                        break
                    if v1 == k1 and v2 == k2:
                        continue
                    b = list(range(N))
                    b[k1] = v1
                    b[k2] = v2
                    # Check if b is a permutation - if so, skip (won't sync)
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
    
    # Strategy 2: Structured random search with local improvement
    random.seed(42)
    
    # Start from best found so far
    if best_a is not None:
        current_a = best_a[:]
        current_b = best_b[:]
    else:
        current_a = a_cycle[:]
        current_b = [0] * N
    
    for iteration in range(4000):
        if time.time() > deadline - 3.0:
            break
        
        # Mutate: change 1-2 transitions in b
        b = current_b[:]
        num_mutations = random.randint(1, 2)
        for _ in range(num_mutations):
            idx = random.randint(0, N-1)
            b[idx] = random.randint(0, N-1)
        
        # Occasionally also mutate a
        a = current_a[:]
        if random.random() < 0.3:
            idx = random.randint(0, N-1)
            a[idx] = random.randint(0, N-1)
        
        if try_automaton(a, b):
            current_a = a[:]
            current_b = b[:]
            # Local search: try single-transition improvements in both a and b
            improved = True
            while improved and time.time() < deadline - 3.0:
                improved = False
                # Try improving b
                for i in range(N):
                    for new_val in range(N):
                        if new_val == current_b[i]:
                            continue
                        old_val = current_b[i]
                        current_b[i] = new_val
                        if try_automaton(current_a, current_b):
                            improved = True
                            break
                        current_b[i] = old_val
                    if improved:
                        break
                # Try improving a
                for i in range(N):
                    for new_val in range(N):
                        if new_val == current_a[i]:
                            continue
                        old_val = current_a[i]
                        current_a[i] = new_val
                        if try_automaton(current_a, current_b):
                            improved = True
                            break
                        current_a[i] = old_val
                    if improved:
                        break
    
    # Strategy 3: Try different 'a' structures
    if time.time() < deadline - 10.0:
        # Try a = identity, b = random non-permutation
        a_id = list(range(N))
        for _ in range(1500):
            if time.time() > deadline - 3.0:
                break
            b = [random.randint(0, N-1) for _ in range(N)]
            if len(set(b)) == N:
                b[random.randint(0, N-1)] = b[random.randint(0, N-1)]
            try_automaton(a_id, b)
        
        # Try a = reversal
        a_rev = [N - 1 - i for i in range(N)]
        for _ in range(1500):
            if time.time() > deadline - 3.0:
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
