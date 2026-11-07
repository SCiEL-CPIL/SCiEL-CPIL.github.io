import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    base = 210
    max_value = 10210
    max_size = 100
    
    # Sieve primes up to max_value
    sieve = [True] * (max_value + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_value**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_value + 1, i):
                sieve[j] = False
    primes = [i for i in range(max_value + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute square-free bitsets using SPF
    spf = list(range(max_value + 1))
    for i in range(2, int(max_value**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_value + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    sqf_bits = [0] * (max_value + 1)
    for n in range(211, max_value + 1):
        temp = n
        bits = 0
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                bits |= (1 << prime_to_idx[p])
        sqf_bits[n] = bits
        
    target_bits = 0
    for p in [2, 3, 5, 7]:
        target_bits |= (1 << prime_to_idx[p])

    def check_feasible(M):
        """
        Checks if there is a subset of {211, ..., M} 
        whose product * 210 is a square, and size <= max_size.
        Returns the list of integers if found, else None.
        """
        if M < 211:
            return None
            
        basis = {}
        # Map index in {211..M} to the vector and combination
        # We need to track which numbers are used to reconstruct the solution
        # idx: 0..M-211 corresponds to number 211+idx
        
        # To save time, we only insert vectors for 211..M
        for i in range(M - 210):
            n = 211 + i
            v = sqf_bits[n]
            comb = 1 << i
            
            curr_v = v
            curr_comb = comb
            
            while curr_v != 0:
                hb = curr_v.bit_length() - 1
                if hb in basis:
                    b_vec, b_combo = basis[hb]
                    curr_v ^= b_vec
                    curr_comb ^= b_combo
                else:
                    basis[hb] = (curr_v, curr_comb)
                    break
        
        # Check if target is in span
        t_curr = target_bits
        t_comb = 0
        
        while t_curr != 0:
            hb = t_curr.bit_length() - 1
            if hb in basis:
                b_vec, b_combo = basis[hb]
                t_curr ^= b_vec
                t_comb ^= b_combo
            else:
                return None
        
        if t_curr == 0:
            sol_mask = t_comb
            sol_count = bin(sol_mask).count('1')
            if 1 <= sol_count <= max_size:
                result = []
                for i in range(M - 210):
                    if (sol_mask >> i) & 1:
                        result.append(211 + i)
                # Verify
                prod = base
                for x in result:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return result
        return None

    low = 211
    high = max_value
    best_result = None
    
    while low <= high:
        if time.time() > deadline:
            break
            
        mid = (low + high) // 2
        result = check_feasible(mid)
        
        if result is not None:
            best_result = result
            high = mid - 1
        else:
            low = mid + 1
            
    if best_result:
        return list(best_result)
        
    # Fallback: Linear scan from 211 up to deadline
    # This handles the case where binary search missed it due to weight issues
    # or if the feasible M is very small and binary search logic was tricky
    # But binary search should find the smallest M where Gaussian solution is light.
    # If no solution found, try linear scan for small M?
    # Actually, if check_feasible returns None, it means either no solution or heavy solution.
    # If we found a solution at mid, we go lower.
    # If we didn't, we go higher.
    # This finds the smallest M where a LIGHT solution exists.
    
    # If best_result is None, try a linear scan from 211 to see if we missed anything
    # (e.g. if the first feasible M had a heavy solution, but a later one had a light one,
    # binary search would find the later one. But what if the FIRST feasible M had a light solution
    # but we skipped it? No, check_feasible returns the solution if it's light.)
    
    # So if best_result is None, it means no M in [211, 10210] yielded a light solution via this method.
    # This is unlikely.
    
    return [840]
