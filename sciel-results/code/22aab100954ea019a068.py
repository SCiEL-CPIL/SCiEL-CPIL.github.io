import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N_MIN = 211
    # We don't know the upper bound, but it's likely small. 
    # Let's assume it's within 10210.
    
    # Target: sf(210)
    # 210 = 2 * 3 * 5 * 7
    # We need to represent these as bitsets.
    # First, we need a mapping from prime to bit index.
    # Since we are iterating M upwards, we can dynamically add primes.
    # However, for bitset operations, it's easier to have a fixed mapping if the range is known.
    # Primes up to 10210 are ~1000 primes. Python ints handle this fine.
    
    # Let's precompute primes up to 10210
    limit = 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
            
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Function to get SF bitset of x
    def get_sf_bitset(x):
        temp = x
        b = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    b |= (1 << prime_to_idx[p])
        if temp > 1:
            b |= (1 << prime_to_idx[temp])
        return b

    target = get_sf_bitset(210)
    
    basis = [] # List of tuples (pivot_bit, vec, mask)
    # pivot_bit is the index of the highest set bit in vec
    
    # We iterate M from N_MIN upwards
    # mask bit i corresponds to number N_MIN + i
    
    for M in range(N_MIN, limit + 1):
        if time.time() > deadline:
            break
            
        v = get_sf_bitset(M)
        idx = M - N_MIN
        m = 1 << idx
        
        # Reduce v using existing basis
        # We should process basis in order of decreasing pivot to ensure echelon form
        # But since we add new pivots, we can just iterate. 
        # Actually, to maintain reduced row echelon or just row echelon, 
        # we need to be careful. 
        # Standard approach: 
        # 1. Reduce v by all existing basis vectors.
        # 2. If v != 0, add to basis.
        
        current_v = v
        current_m = m
        
        for b_vec, b_mask, b_pivot in basis:
            if (current_v >> b_pivot) & 1:
                current_v ^= b_vec
                current_m ^= b_mask
        
        if current_v != 0:
            # Find pivot of current_v
            pivot = current_v.bit_length() - 1
            # Insert into basis
            # We should keep basis sorted by pivot descending for efficient reduction?
            # Or just append. Since we reduce against ALL basis vectors, order doesn't strictly matter for correctness of "is 0", 
            # but for performance, reducing with higher pivots first is better.
            basis.append((current_v, current_m, pivot))
            # Sort to keep highest pivot first? 
            # Actually, if we always reduce against all, the order in the list doesn't matter for the final result of reduction, 
            # only that we visit all. 
            # However, if we want to check target membership efficiently later, having them sorted helps.
            # Let's sort every time? No, that's O(B log B). 
            # Given B is small (< 1000), it's fine.
            basis.sort(key=lambda x: x[2], reverse=True)
            
        # Check if target is in span
        w = target
        res_mask = 0
        for b_vec, b_mask, b_pivot in basis:
            if (w >> b_pivot) & 1:
                w ^= b_vec
                res_mask ^= b_mask
        
        if w == 0:
            # Found solution
            selected = []
            # res_mask has bits set for indices 0..M-N_MIN
            for i in range(idx + 1):
                if (res_mask >> i) & 1:
                    selected.append(N_MIN + i)
            
            # Verify
            prod = 210
            for x in selected:
                prod *= x
            s = math.isqrt(prod)
            if s * s == prod:
                return selected
            else:
                # Should not happen if logic is correct, but fallback
                # Try to find a simpler solution if this fails? 
                # Unlikely.
                pass
                
    # Fallback if not found (should not happen)
    return [840]
