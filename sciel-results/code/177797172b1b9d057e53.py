import time
import random

def construct():
    start_time = time.time()
    limit = 55.0 # Safety margin for 60s limit
    
    n = 256
    
    # Weights: 1/i
    weights = [1.0 / i for i in range(1, n + 1)]
    
    # Initial Solution: Greedy by weight (small index first)
    # We want to maximize sum(1/x), so we prefer small x.
    S_bits = 0
    S_set = set()
    
    # Greedy: iterate 1..n, add if possible
    for x in range(1, n + 1):
        if can_add(x, S_bits):
            S_bits |= (1 << x)
            S_set.add(x)
            
    best_value = sum(1.0 / x for x in S_set)
    best_solution = list(S_set)
    
    # Local Search
    improved = True
    while improved and time.time() - start_time < limit - 5:
        improved = False
        
        # 1. Try to add any element
        for x in range(1, n + 1):
            if not (S_bits & (1 << x)):
                if can_add(x, S_bits):
                    S_bits |= (1 << x)
                    S_set.add(x)
                    best_value += 1.0 / x
                    improved = True
                    
        # 2. Try to swap one element with another
        # For each u in S, try removing it and adding v not in S
        # If 1/v > 1/u, it's a potential improvement
        # But we must check 3-AP freeness.
        
        # Optimization: Only try swaps if 1/v > 1/u? 
        # Yes, because if 1/v <= 1/u, the weight doesn't increase, 
        # and we might lose flexibility? 
        # Actually, swapping a large weight for a small weight is bad.
        # Swapping a small weight for a large weight is good.
        # So we only need to consider v < u (since 1/v > 1/u).
        
        for u in list(S_set):
            # Temporarily remove u
            S_bits_without_u = S_bits & ~(1 << u)
            S_set_without_u = S_set - {u}
            
            # Try adding v
            # v must be < u to improve weight
            for v in range(1, u):
                if S_bits_without_u & (1 << v):
                    continue
                if can_add(v, S_bits_without_u):
                    # Check if this swap improves value
                    # Value change: -1/u + 1/v
                    # Since v < u, 1/v > 1/u, so it always improves.
                    # Just need to check if it's a valid 3-AP free set.
                    # We already checked can_add(v, S \ {u}).
                    # So S \ {u} U {v} is valid.
                    
                    # Accept swap
                    S_bits = S_bits_without_u | (1 << v)
                    S_set.remove(u)
                    S_set.add(v)
                    best_value += (1.0 / v) - (1.0 / u)
                    improved = True
                    # Break out of v loop? 
                    # We removed u. We should restart the u loop or continue?
                    # If we accept a swap, the set changed. 
                    # It's safer to restart the outer loop or just mark improved and break.
                    # Let's break out of v loop and restart u loop? 
                    # Or just continue with the new set?
                    # If we continue, we might swap v out immediately if it's bad?
                    # But we only swap to improve.
                    # Let's break out of the inner loop and restart the outer loop to be safe.
                    break
            if improved:
                # Restart the u loop? 
                # The set changed. The list(S_set) is stale.
                # It's better to break out of the while loop and restart the whole local search phase?
                # Or just break the u loop and let the while loop restart.
                break
                
    # 3. Try 2-for-1 swaps? (Remove 2, Add 1? No, weight drops. Remove 1, Add 2? Weight might increase if 1/v1 + 1/v2 > 1/u)
    # This is more complex. Given time, 1-for-1 swaps are likely sufficient to reach a good local optimum.
    # The greedy start is strong.
    
    if time.time() - start_time < limit - 2:
         # Try a few random restarts or perturbations?
         # Given the deterministic nature of the greedy start, we might be stuck in a local optimum.
         # But the swap search is quite thorough.
         pass

    return list(S_set)

def can_add(x, S_bits):
    """
    Check if x can be added to S (represented by S_bits) without creating a 3-AP.
    x is 1-indexed.
    """
    # Iterate through all y in S
    # y is 1-indexed. Bit index is y.
    # We need to check if there exists y in S such that:
    # 1. y, x, z form AP -> z = 2x - y. Check if z in S.
    # 2. x, y, z form AP -> z = 2y - x. Check if z in S.
    # 3. x, z, y form AP -> z = (x+y)/2. Check if z in S and x+y is even.
    
    # Efficiently iterate bits
    temp = S_bits
    while temp:
        # Get index of lowest set bit
        # y is the index (1-based)
        # (temp & -temp) gives the lowest bit value.
        # log2 of that value is the 0-based index.
        # We want 1-based index y.
        low_bit = temp & -temp
        y = low_bit.bit_length()  # This is 1-based index because 1<<1 is 2, bit_length is 2? No.
        # 1<<1 = 2 (binary 10). bit_length is 2. Index is 1.
        # 1<<0 = 1 (binary 1). bit_length is 1. Index is 0.
        # So y = low_bit.bit_length() - 1?
        # Let's check:
        # If S has 1, bit is 1<<1 = 2. bit_length is 2. y should be 1.
        # If S has 2, bit is 1<<2 = 4. bit_length is 3. y should be 2.
        # So y = low_bit.bit_length() - 1.
        
        y = low_bit.bit_length() - 1
        
        # Check 1: y, x, z. z = 2x - y
        z = 2 * x - y
        if 1 <= z <= 256:
            if S_bits & (1 << z):
                return False
                
        # Check 2: x, y, z. z = 2y - x
        z = 2 * y - x
        if 1 <= z <= 256:
            if S_bits & (1 << z):
                return False
                
        # Check 3: x, z, y. z = (x+y)/2
        if (x + y) % 2 == 0:
            z = (x + y) // 2
            if 1 <= z <= 256:
                if S_bits & (1 << z):
                    return False
        
        temp &= temp - 1 # Clear lowest bit
        
    return True
