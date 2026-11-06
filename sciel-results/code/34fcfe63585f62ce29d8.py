import time
import random
from math import gcd

def construct():
    N = 128
    # Precompute LCMs and bad masks
    # bad_mask[i][j] is a bitmask of k such that lcm(i,j) == lcm(i,k) == lcm(j,k)
    # Indices 0..N-1 represent 1..N
    bad_mask = [[0] * N for _ in range(N)]
    
    # Precompute LCMs
    # lcm(i, j) for 1..N
    # Note: i, j are 1-based in logic, 0-based in arrays
    lcm_val = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            a = i + 1
            b = j + 1
            lcm_val[i][j] = (a * b) // gcd(a, b)
            
    # Compute bad masks
    for i in range(N):
        for j in range(i + 1, N):
            L = lcm_val[i][j]
            mask = 0
            for k in range(N):
                if k == i or k == j:
                    continue
                # Check if lcm(i,k) == L and lcm(j,k) == L
                # We can compute lcm on the fly or precompute all lcms
                # Precomputing all lcms is 128*128 = 16k, fast
                pass
            
    # Let's precompute all lcms first to make bad mask computation fast
    lcm_full = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            a = i + 1
            b = j + 1
            lcm_full[i][j] = (a * b) // gcd(a, b)
            
    for i in range(N):
        for j in range(i + 1, N):
            L = lcm_full[i][j]
            mask = 0
            for k in range(N):
                if k == i or k == j:
                    continue
                if lcm_full[i][k] == L and lcm_full[j][k] == L:
                    mask |= (1 << k)
            bad_mask[i][j] = mask
            bad_mask[j][i] = mask

    def get_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            bits.append(idx)
            mask ^= lsb
        return bits

    def greedy_and_local_search(order):
        S = set()
        # count[k] tracks how many pairs in S have k in their bad mask
        counts = [0] * N
        blocked_mask = 0
        
        def add_element(x):
            nonlocal blocked_mask
            S.add(x)
            for z in list(S):
                if z == x:
                    continue
                m = bad_mask[x][z]
                bits = get_bits(m)
                for k in bits:
                    counts[k] += 1
                    if counts[k] == 1:
                        blocked_mask |= (1 << k)
            
        def remove_element(x):
            nonlocal blocked_mask
            S.remove(x)
            for z in list(S):
                if z == x:
                    continue
                m = bad_mask[x][z]
                bits = get_bits(m)
                for k in bits:
                    counts[k] -= 1
                    if counts[k] == 0:
                        blocked_mask &= ~(1 << k)
                        
        # Greedy Phase
        for x in order:
            if not (blocked_mask & (1 << x)):
                add_element(x)
                
        # Local Search Phase
        improved = True
        while improved:
            improved = False
            # Try to add any element
            for y in range(N):
                if y not in S and not (blocked_mask & (1 << y)):
                    add_element(y)
                    improved = True
                    break # Restart loop to re-evaluate
            
            if improved:
                continue
                
            # Try swaps: remove x, add y
            # To speed up, we can iterate x in S, y not in S
            # Check if S \ {x} + {y} is valid
            # This requires checking if y is not blocked by S \ {x}
            # blocked_mask(S \ {x}) is not directly available.
            # We can compute it temporarily.
            # Cost: O(|S| * |C| * |S| * avg_bad_size)
            # |S| ~ 80, |C| ~ 48. 80*48 = 3840 swaps.
            # Each swap check: remove x (O(|S|)), check y, add x back (O(|S|)).
            # Total ~ 3840 * 160 * 2 ~ 1M ops. Fast enough.
            
            for x in list(S):
                # Temporarily remove x
                remove_element(x)
                for y in range(N):
                    if y in S:
                        continue
                    if not (blocked_mask & (1 << y)):
                        # Valid swap
                        add_element(y)
                        improved = True
                        break
                if improved:
                    break
                # Put x back
                add_element(x)
                
        return S

    best_set = set()
    deadline = time.time() + 50.0
    
    # Run multiple random trials
    while time.time() < deadline:
        order = list(range(N))
        random.shuffle(order)
        S = greedy_and_local_search(order)
        if len(S) > len(best_set):
            best_set = S
            
    return list(best_set)
