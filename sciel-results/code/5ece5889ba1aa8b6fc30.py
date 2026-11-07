"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _isqrt(n):
    if n < 0:
        raise ValueError
    if n == 0:
        return 0
    return math.isqrt(n)

def construct():
    n_base = 210
    max_size = 100
    max_value = 10210

    # Primes up to sqrt(max_value) ~ 101
    # Use a simple sieve
    limit = 150
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    # Parity vector of 210 = 2 * 3 * 5 * 7
    # Target: each of 2,3,5,7 appears to odd power
    target_primes = {2, 3, 5, 7}

    def factor_parity(n):
        """Return frozenset of primes dividing n to an odd power."""
        result = set()
        temp = n
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                result.add(p)
        if temp > 1:
            result.add(temp)
        return frozenset(result)

    def verify(subset):
        """Verify that product(subset) * 210 is a perfect square."""
        prod = 210
        for x in subset:
            prod *= x
        s = _isqrt(prod)
        return s * s == prod

    def find_subset_for_m(m):
        """Find a subset of integers in (210, m] with XOR = target, size <= 100.
        Returns the subset or None."""
        if m <= 210:
            return None

        candidates = list(range(211, m + 1))
        
        # Compute parity vectors
        parity_vectors = []
        for num in candidates:
            pf = factor_parity(num)
            parity_vectors.append(pf)
        
        # Collect all primes that appear
        all_primes_set = set()
        for pv in parity_vectors:
            all_primes_set.update(pv)
        all_primes_set.update(target_primes)
        
        prime_list = sorted(all_primes_set)
        prime_to_idx = {p: i for i, p in enumerate(prime_list)}
        num_primes = len(prime_list)
        
        def to_bitset(fset):
            bits = 0
            for p in fset:
                if p in prime_to_idx:
                    bits |= (1 << prime_to_idx[p])
            return bits

        target_bits = to_bitset(target_primes)
        
        # Build matrix: each row is a parity vector, last row is target
        mat = [to_bitset(v) for v in parity_vectors]
        mat.append(target_bits)
        
        n_rows = len(mat)
        
        # Track combinations: comb[r] is a bitset of which original vectors (0..n-1) combine to form row r
        comb = [1 << i for i in range(len(parity_vectors))]
        comb.append(0)  # Target row starts with empty combination
        
        # Gaussian elimination over GF(2)
        rank = 0
        for col in range(num_primes):
            # Find pivot row
            pivot = -1
            for r in range(rank, n_rows):
                if (mat[r] >> col) & 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            # Swap pivot with current rank row
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            comb[rank], comb[pivot] = comb[pivot], comb[rank]
            # Eliminate all other rows
            for r in range(n_rows):
                if r != rank and ((mat[r] >> col) & 1):
                    mat[r] ^= mat[rank]
                    comb[r] ^= comb[rank]
            rank += 1
        
        # Check if target row reduced to zero (target is in span)
        if mat[-1] != 0:
            return None
        
        # Extract subset indices from comb[-1]
        c = comb[-1]
        subset_indices = [i for i in range(len(parity_vectors)) if (c >> i) & 1]
        
        if len(subset_indices) == 0:
            return None
        
        if len(subset_indices) > max_size:
            return None
        
        subset = [candidates[i] for i in subset_indices]
        
        # Verify
        if not verify(subset):
            return None
        
        return subset

    # Binary search on M to find minimal bound
    lo = 211
    hi = max_value
    best_result = None
    best_m = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        result = find_subset_for_m(mid)
        if result is not None:
            best_result = result
            best_m = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_result is None:
        # Fallback: use a known valid subset
        # 210 * 420 * 840 = 210 * 2*210 * 4*210 = 2^6 * 3^3 * 5^3 * 7^3
        # Hmm, let me think of a simpler fallback
        # 210 * 420 = 210^2 * 2 = not a square
        # 210 * 420 * 840 * 1260 = 210^4 * 2*4*6 = 210^4 * 48 = not square
        # Let's just return [420, 840, 1260, 1680, 2100, 2520, 2940, 3360, 3780, 4200]
        # Actually, let me construct a guaranteed valid set
        # If we pick numbers of the form 210*k where k is a square, their product times 210
        # = 210 * prod(210*k_i) = 210^(n+1) * prod(k_i)
        # For n=1: 210^2 * k_1, which is a square if k_1 is a square.
        # So [420] alone: 210*420 = 210^2 * 2, not square.
        # [840]: 210*840 = 210^2 * 4 = (210*2)^2 = 420^2. Yes!
        return [840]
    
    # Ensure types are correct
    return list(best_result)
# EVOLVE-BLOCK-END
