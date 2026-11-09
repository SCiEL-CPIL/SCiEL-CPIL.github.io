import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 30
    half = 15
    
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    weights = [L // i for i in range(1, n + 1)]
    
    # Weights for first half and second half
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Helper to pack/unpack coefficients
    def pack_coeffs(coeffs):
        # coeffs is a list/tuple of 15 values in {-1, 0, 1}
        # Map -1->0, 0->1, 1->2
        val = 0
        for c in coeffs:
            val = val * 3 + (c + 1)
        return val
    
    def unpack_coeffs(packed, length):
        coeffs = []
        for _ in range(length):
            coeffs.append(packed % 3 - 1)
            packed //= 3
        return coeffs

    # Generate all sums for first half
    # Using iterative approach for speed
    sums1 = []
    # Start with 0 coeffs
    current_coeffs = [0] * half
    current_sum = 0
    
    # We can use a stack or recursion. Iterative with base-3 counter is fast.
    # Total combinations: 3^15
    total_combos = 3 ** half
    
    # Pre-compute powers of 3? No, just iterate.
    # Efficient way:
    # Instead of updating sum from scratch, we can use a recursive generator, 
    # but Python recursion limit and overhead might be an issue.
    # Let's use a simple loop with a base-3 counter.
    
    # Actually, a recursive function is cleaner and often fast enough for depth 15.
    # But let's try to optimize.
    
    def generate_sums(weights, length):
        results = []
        # Recursive helper
        def rec(idx, current_sum, current_coeffs):
            if time.time() > deadline:
                return
            if idx == length:
                results.append((current_sum, pack_coeffs(current_coeffs)))
                return
            for val in [-1, 0, 1]:
                current_coeffs[idx] = val
                rec(idx + 1, current_sum + val * weights[idx], current_coeffs)
        
        rec(0, 0, [0] * length)
        return results

    # This recursion might be slow due to function calls.
    # 3^15 = 14M calls. Python does ~10M-50M simple ops/sec.
    # 14M function calls might take 5-10 seconds.
    # Let's try to make it iterative or use a more efficient generation.
    
    # Alternative: Use numpy? No, standard lib preferred for this logic unless needed.
    # Let's stick to recursion but optimize.
    
    # Better: Iterative expansion
    def generate_sums_iter(weights, length):
        # Start with list of (sum, packed_coeffs)
        # Initially: [(0, 0)]
        sums = [(0, 0)]
        for i in range(length):
            w = weights[i]
            new_sums = []
            for s, p in sums:
                # Append -1, 0, 1
                # Packing: p is base-3 number of previous coeffs.
                # New coeff is at position i.
                # If we build from left to right, the new digit is the least significant?
                # My pack function: val = val * 3 + (c+1).
                # So c[0] is most significant.
                # If we iterate i from 0 to length-1:
                # At step i, we have packed c[0]...c[i-1].
                # We add c[i].
                # new_packed = p * 3 + (c[i] + 1)
                
                new_sums.append((s - w, p * 3 + 0)) # c=-1 -> 0
                new_sums.append((s, p * 3 + 1))     # c=0 -> 1
                new_sums.append((s + w, p * 3 + 2)) # c=1 -> 2
            sums = new_sums
            if time.time() > deadline:
                break
        return sums

    # This iterative approach creates lists of size 3^i.
    # 3^15 is 14M.
    # Memory: 14M tuples of (int, int).
    # This might be heavy but let's try.
    
    sums1 = generate_sums_iter(w1, half)
    if time.time() > deadline:
        # Fallback
        return [0]*30
    
    sums2 = generate_sums_iter(w2, half)
    if time.time() > deadline:
        return [0]*30

    # Sort by sum
    sums1.sort(key=lambda x: x[0])
    sums2.sort(key=lambda x: x[0])
    
    # Two-pointer search
    # We want to minimize |s1 + s2| such that s1 + s2 != 0.
    
    i = 0
    j = len(sums2) - 1
    best_diff = float('inf')
    best_s1 = None
    best_s2 = None
    best_p1 = None
    best_p2 = None
    
    len1 = len(sums1)
    len2 = len(sums2)
    
    while i < len1 and j >= 0:
        s1, p1 = sums1[i]
        s2, p2 = sums2[j]
        diff = s1 + s2
        
        if diff == 0:
            # We hit zero. We need to check neighbors.
            # The next best candidates are (i+1, j) and (i, j-1).
            # Also (i-1, j) and (i, j+1) are already considered or will be?
            # Since s1 is increasing and s2 is decreasing (as j decreases),
            # s1[i+1] + s2[j] > 0
            # s1[i] + s2[j-1] < 0
            
            # Check i+1, j
            if i + 1 < len1:
                ns1, np1 = sums1[i+1]
                ndiff = ns1 + s2
                if ndiff != 0 and abs(ndiff) < best_diff:
                    best_diff = abs(ndiff)
                    best_s1 = ns1
                    best_s2 = s2
                    best_p1 = np1
                    best_p2 = p2
            
            # Check i, j-1
            if j - 1 >= 0:
                ns2, np2 = sums2[j-1]
                ndiff = s1 + ns2
                if ndiff != 0 and abs(ndiff) < best_diff:
                    best_diff = abs(ndiff)
                    best_s1 = s1
                    best_s2 = ns2
                    best_p1 = p1
                    best_p2 = np2
            
            # Move pointers to explore other regions
            # We can move both? Or just one?
            # If we move i, s1 increases, sum increases.
            # If we move j, s2 decreases, sum decreases.
            # We want to find other small sums.
            # Standard two pointer for closest sum to 0:
            # If sum == 0, we've found the minimum absolute sum (0).
            # But we need non-zero.
            # So we look for the smallest non-zero.
            # The candidates are the immediate neighbors in the sorted order of sums.
            # Since we are at 0, the next smallest absolute sums will be found by moving i or j.
            # Let's move i forward and j backward.
            i += 1
            j -= 1
            continue
        
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_s1 = s1
            best_s2 = s2
            best_p1 = p1
            best_p2 = p2
        
        if diff < 0:
            # Sum is too negative, need larger s1
            i += 1
        else:
            # Sum is too positive, need smaller s2
            j -= 1
        
        if time.time() > deadline:
            break

    # Reconstruct coefficients
    if best_p1 is not None and best_p2 is not None:
        c1 = unpack_coeffs(best_p1, half)
        c2 = unpack_coeffs(best_p2, half)
        return c1 + c2
    
    # Fallback
    return [0]*30
