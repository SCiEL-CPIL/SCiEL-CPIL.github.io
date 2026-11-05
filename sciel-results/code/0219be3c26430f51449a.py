import time
import random
from collections import deque

N = 12

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton with bitmasks for efficiency.
    Optimized for speed using list-based queue and array-based visited.
    """
    all_states = (1 << N) - 1
    
    # Precompute transition tables for bitmask application
    # We can precompute the effect of a and b on every possible subset?
    # There are 2^12 = 4096 subsets.
    # Precomputing trans_a[mask] and trans_b[mask] is very fast.
    
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    # Initialize
    for mask in range(1 << N):
        if mask == 0:
            continue
        # Compute next state for a
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            res_a |= (1 << a_trans[i])
            res_b |= (1 << b_trans[i])
            m ^= lsb
        trans_a[mask] = res_a
        trans_b[mask] = res_b

    start = all_states
    
    # Check if already singleton
    if (start & (start - 1)) == 0:
        return 0

    # BFS
    # visited[mask] = distance
    visited = [-1] * (1 << N)
    visited[start] = 0
    q = [start]
    head = 0
    
    while head < len(q):
        curr = q[head]
        head += 1
        curr_dist = visited[curr]
        
        # Check if singleton
        if (curr & (curr - 1)) == 0:
            return curr_dist
        
        next_a = trans_a[curr]
        if visited[next_a] == -1:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        next_b = trans_b[curr]
        if visited[next_b] == -1:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_score = -1
    best_a = None
    best_b = None
    
    # Helper to update best
    def update(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = list(a)
            best_b = list(b)
        return score

    # Strategy: Systematic search over structured families
    # It is known that optimal automata often have 'a' as a cycle.
    
    # 1. 'a' is a full cycle (shift by 1)
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Family 1: b is a permutation.
    # Specifically, try b as a cycle of length N-1 fixing one state, or similar.
    # Also try b as identity with defects.
    
    # Sub-family 1a: b is identity with 1 defect
    for k in range(N):
        for j in range(N):
            if j == k:
                continue
            b = [i for i in range(N)]
            b[k] = j
            if time.time() > deadline:
                return [best_a, best_b]
            update(a_cycle, b)
            
    # Sub-family 1b: b is a single cycle of length N-1 fixing one state
    # b[fixed] = fixed
    # b[rest] = cycle
    for fixed in range(N):
        others = [i for i in range(N) if i != fixed]
        # Generate all cycles on 'others'
        # To avoid duplicates, fix the first element of the cycle to be the smallest in 'others'
        # But we just need to find high scores.
        # Number of cycles on N-1 elements is (N-2)!. For N=12, 10! = 3,628,800.
        # This might be too slow if done naively.
        # Let's sample or use a smarter approach.
        # Actually, the optimal structure for Pinsker is often:
        # a: 0->1->...->N-1->0
        # b: 0->0, 1->2, ..., N-1->1 (cycle on 1..N-1)
        # Let's test this specific structure first.
        b_pinsker = [0] * N
        b_pinsker[0] = 0
        for i in range(1, N):
            b_pinsker[i] = (i + 1) % N
            if b_pinsker[i] == 0:
                b_pinsker[i] = 1
        # Correct cycle: 1->2, 2->3, ..., N-2->N-1, N-1->1
        b_pinsker = list(range(N))
        b_pinsker[0] = 0
        for i in range(1, N):
            b_pinsker[i] = (i + 1) % N
            if b_pinsker[i] == 0:
                b_pinsker[i] = 1
        # Let's just construct it properly
        b_pinsker = [0] * N
        b_pinsker[0] = 0
        for i in range(1, N):
            if i == N - 1:
                b_pinsker[i] = 1
            else:
                b_pinsker[i] = i + 1
        update(a_cycle, b_pinsker)
        
    # Sub-family 1c: Random permutations for b
    random.seed(42)
    for _ in range(2000):
        if time.time() > deadline:
            return [best_a, best_b]
        b = list(range(N))
        random.shuffle(b)
        # Ensure it's not the identity (which might be low score)
        # But we check score anyway.
        update(a_cycle, b)

    # 2. 'a' is a shift by d
    for d in range(1, N):
        a_shift = [(i + d) % N for i in range(N)]
        # Try b as identity with 1 defect
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                b = [i for i in range(N)]
                b[k] = j
                if time.time() > deadline:
                    return [best_a, best_b]
                update(a_shift, b)
                
        # Try b as Pinsker-like cycle fixing one state
        for fixed in range(N):
            b = [0] * N
            b[fixed] = fixed
            others = [i for i in range(N) if i != fixed]
            # Create a cycle on 'others'
            # Simple cycle: others[0] -> others[1] -> ... -> others[-1] -> others[0]
            for i in range(len(others)):
                b[others[i]] = others[(i + 1) % len(others)]
            if time.time() > deadline:
                return [best_a, best_b]
            update(a_shift, b)

    # 3. Local Search / Hill Climbing
    # Start from best found so far
    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N
        update(best_a, best_b)
        
    current_a = best_a
    current_b = best_b
    current_score = best_score
    
    # Try to mutate b
    for _ in range(5000):
        if time.time() > deadline:
            return [best_a, best_b]
        
        b_new = list(current_b)
        # Mutate 1 or 2 positions
        num_mut = 1 if random.random() < 0.5 else 2
        for _ in range(num_mut):
            idx = random.randint(0, N-1)
            val = random.randint(0, N-1)
            b_new[idx] = val
            
        score = calculate_threshold(current_a, b_new)
        if score > current_score:
            current_b = b_new
            current_score = score
            if score > best_score:
                best_score = score
                best_a = list(current_a)
                best_b = list(current_b)
        
        # Occasionally mutate a
        if random.random() < 0.1:
            a_new = list(current_a)
            idx = random.randint(0, N-1)
            val = random.randint(0, N-1)
            a_new[idx] = val
            score = calculate_threshold(a_new, current_b)
            if score > current_score:
                current_a = a_new
                current_score = score
                if score > best_score:
                    best_score = score
                    best_a = list(current_a)
                    best_b = list(current_b)

    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
