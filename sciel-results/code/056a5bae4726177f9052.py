import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit
    N_MAX = 4096
    MAX_SUM = 3 * N_MAX
    best_set = []
    best_size = 0

    def can_add(S, x, sum_flag):
        """Check if x can be added to S maintaining B_3[1] property."""
        # Check 3*x
        if sum_flag[3 * x]:
            return False
        
        # Check 2*x + s for each s in S
        for s in S:
            if sum_flag[2 * x + s]:
                return False
        
        # Check x + s1 + s2 for s1 <= s2 in S
        k = len(S)
        for i in range(k):
            s1 = S[i]
            base = x + s1
            for j in range(i, k):
                if sum_flag[base + S[j]]:
                    return False
        return True

    def add_x(S, x, sum_flag):
        """Add x to S and update sum_flag."""
        sum_flag[3 * x] = True
        for s in S:
            sum_flag[2 * x + s] = True
        k = len(S)
        for i in range(k):
            s1 = S[i]
            base = x + s1
            for j in range(i, k):
                sum_flag[base + S[j]] = True
        S.append(x)

    # Strategy 1: Random restarts with greedy selection
    # Try different random seeds to explore the solution space
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart)
        # Random permutation of candidates
        candidates = list(range(1, N_MAX + 1))
        rng.shuffle(candidates)
        
        S = []
        sum_flag = bytearray(MAX_SUM + 1)  # Use bytearray for memory efficiency
        
        for x in candidates:
            if len(S) >= 80:
                break
            
            if can_add(S, x, sum_flag):
                add_x(S, x, sum_flag)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
            if best_size == 80:
                return best_set

    # Strategy 2: Start from a geometric skeleton and fill in
    # Powers of 4 give valid B_3 sets: {1, 4, 16, 64, ...}
    # We can use this as a base and try to fill with additional elements
    
    for base_power in [3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Create a geometric base
        base = [base_power ** i for i in range(10) if base_power ** i <= N_MAX]
        
        S = list(base)
        sum_flag = bytearray(MAX_SUM + 1)
        
        # Initialize sum_flag with the base set
        for i, a in enumerate(S):
            for j in range(i, len(S)):
                for k in range(j, len(S)):
                    sum_flag[a + S[j] + S[k]] = True
        
        # Try to fill with random elements
        rng = random.Random(1000 + base_power)
        remaining = list(set(range(1, N_MAX + 1)) - set(S))
        rng.shuffle(remaining)
        
        for x in remaining:
            if len(S) >= 80:
                break
            if can_add(S, x, sum_flag):
                add_x(S, x, sum_flag)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            if best_size == 80:
                return best_set

    # Strategy 3: Local improvement - try to replace elements to free up space
    if best_size > 0 and time.time() < deadline - 10:
        # Try random restarts with a focus on smaller numbers first
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            rng = random.Random(5000 + attempt)
            # Bias towards smaller numbers for initial selection
            weights = [1.0 / (i + 1) for i in range(N_MAX)]
            candidates = list(range(1, N_MAX + 1))
            rng.shuffle(candidates)
            
            S = []
            sum_flag = bytearray(MAX_SUM + 1)
            
            for x in candidates:
                if len(S) >= 80:
                    break
                if can_add(S, x, sum_flag):
                    add_x(S, x, sum_flag)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                if best_size == 80:
                    return best_set

    return best_set
