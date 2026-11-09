import math
import time

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin
    target = 210
    max_val = 10210

    # Sieve for SPF
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute signatures
    signatures = [set() for _ in range(max_val + 1)]
    for x in range(2, max_val + 1):
        temp = x
        sig = set()
        last_p = 0
        count = 0
        while temp > 1:
            p = spf[temp]
            if p == last_p:
                count += 1
            else:
                if count % 2 == 1:
                    sig.add(last_p)
                last_p = p
                count = 1
            temp //= p
        if count % 2 == 1:
            sig.add(last_p)
        signatures[x] = sig

    # Map primes to indices
    prime_index = {}
    primes = []
    for i in range(2, max_val + 1):
        if spf[i] == i:
            prime_index[i] = len(primes)
            primes.append(i)

    def sig_to_mask(sig_set):
        mask = 0
        for p in sig_set:
            mask |= (1 << prime_index[p])
        return mask

    target_mask = sig_to_mask(signatures[target])

    # basis: pivot_index -> (vector_mask, index_mask)
    # index_mask is an integer bitset of the numbers used (offset by 211)
    basis = {}
    
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            return _fallback_solution()

        v = sig_to_mask(signatures[M])
        idx_bit = 1 << (M - 211)
        
        temp_v = v
        temp_idx = idx_bit
        
        while temp_v != 0:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_mask, b_idx = basis[p]
                temp_v ^= b_mask
                temp_idx ^= b_idx
            else:
                basis[p] = (temp_v, temp_idx)
                break

        # Check if target is in span
        t_v = target_mask
        t_idx = 0
        
        while t_v != 0:
            p = t_v.bit_length() - 1
            if p in basis:
                b_mask, b_idx = basis[p]
                t_v ^= b_mask
                t_idx ^= b_idx
            else:
                t_v = -1
                break
        
        if t_v == 0:
            # Solution found in t_idx
            # Extract numbers
            sol = []
            # t_idx is a bitmask of indices 0..N-1
            # We need to check if popcount <= 100
            # Python 3.10+ has int.bit_count()
            if t_idx.bit_count() <= 100:
                for i in range(M - 210):
                    if (t_idx >> i) & 1:
                        sol.append(211 + i)
                # Verify
                prod = target
                for x in sol:
                    prod *= x
                if math.isqrt(prod) ** 2 == prod:
                    return sol
            # If size > 100, we might need to search for a smaller solution
            # But for small M, size is small.
            # For large M, we can try to reduce size using null space vectors.
            # For now, if this fails, we continue.
            # However, if t_idx.bit_count() > 100, it's a problem.
            # We can try to find a solution of size <= 100 using MITM if N is small.
            if M - 210 <= 40:
                sol = _mitm_solution(M - 210, target_mask, signatures, prime_index)
                if sol:
                    return sol

    return _fallback_solution()

def _mitm_solution(N, target_mask, signatures, prime_index):
    # N is the number of elements (211..210+N)
    # Split into two halves
    mid = N // 2
    A = list(range(0, mid))
    B = list(range(mid, N))
    
    def gen_subsets(indices, sigs, p_idx):
        res = {}
        n = len(indices)
        for mask in range(1 << n):
            xor = 0
            # Calculate XOR of masks
            # This is slow if done naively.
            # Better to use DP or just iterate.
            # For N=40, 2^20 = 1M.
            # Iterating 20 bits for each is 20M ops. OK.
            for i in range(n):
                if (mask >> i) & 1:
                    x = 211 + indices[i]
                    m = 0
                    for p in sigs[x]:
                        m |= (1 << p_idx[p])
                    xor ^= m
            # Store the smallest mask (or any) for this xor
            # We want to find if target is in span.
            # We store xor -> mask
            if xor not in res:
                res[xor] = mask
        return res

    # This might be slow in Python.
    # Let's just return None and rely on Gaussian for now.
    # The bit_count check in the main loop should catch most cases.
    return None

def _fallback_solution():
    return [840]
