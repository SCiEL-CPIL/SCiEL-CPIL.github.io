import time
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s
    
    # Step 1: Generate powerful numbers
    # We want a good set of small powerful numbers.
    # Limit L determines how many we get.
    # Count ~ 1.47 * sqrt(L)
    # Let's target L = 10^8 -> ~46000 numbers.
    # We will take the smallest K of them.
    
    L = 10**8
    powerful = set()
    
    # b^3 <= L
    max_b = int(round(L ** (1/3))) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        # a^2 <= L / b^3
        max_a = int(math.isqrt(L // b3))
        for a in range(1, max_a + 1):
            val = (a * a) * b3
            if val <= L:
                powerful.add(val)
            else:
                break
        
    P = sorted(powerful)
    
    # Check time
    if time.time() - start_time > deadline - 30:
         # Fallback to smaller set if generation took too long (unlikely)
         P = P[:5000]
    
    # Step 2: Determine K based on remaining time
    # We estimate time per pair.
    # Let's just pick a K that is safe.
    # 10,000 pairs -> 50M iterations.
    # 20,000 pairs -> 200M iterations.
    
    # Let's try to be adaptive.
    # But simple is better.
    K = 12000
    if len(P) < K:
        K = len(P)
    
    P = P[:K]
    
    # Step 3: Count sums
    counts = {}
    # Use local variables for speed
    P_list = P
    K_val = K
    for i in range(K_val):
        pi = P_list[i]
        for j in range(i, K_val):
            s = pi + P_list[j]
            if s in counts:
                counts[s] += 1
            else:
                counts[s] = 1
        
        # Time check inside loop
        if i % 1000 == 0 and time.time() - start_time > deadline - 20:
            break
            
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Find max
    max_count = 0
    best_N = 0
    for s, c in counts.items():
        if c > max_count:
            max_count = c
            best_N = s
            
    # Step 4: Reconstruct pairs for best_N
    pairs = []
    for i in range(K_val):
        a = P_list[i]
        if 2 * a > best_N:
            break
        b = best_N - a
        # Check if b is in P and b >= a
        # Since P is sorted, we can just check if b is in the set of P
        # But we need to ensure b is in P_list (the truncated list)
        # Actually, b might be larger than P_list[-1]?
        # If a is small, b is large.
        # If b > P_list[-1], it's not in our candidate set.
        # But we generated sums only from P_list.
        # So if a+b = best_N was counted, b MUST be in P_list.
        # So we can just check if b is in the set of P_list.
        
        # To make this O(1), create a set
        # We can create the set once.
        
    # Optimization: Create set of P_list for O(1) lookup
    P_set = set(P_list)
    pairs = []
    for a in P_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in P_set:
            pairs.append([a, b])
            
    # Ensure distinct pairs and a<=b
    # The loop above ensures a<=b and distinct a.
    # Since a and b are unique in the set?
    # Powerful numbers are unique values.
    # So pairs are distinct.
    
    return {'N': best_N, 'pairs': pairs}
