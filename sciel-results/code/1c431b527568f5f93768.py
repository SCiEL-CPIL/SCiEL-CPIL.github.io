import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 210
    MAX_VAL = 10210
    
    # Sieve primes up to MAX_VAL
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]
    
    primes = sieve(MAX_VAL)
    D = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute vector (bitmask of primes with odd exponent) for each number
    vecs = [0] * (MAX_VAL + 1)
    for x in range(2, MAX_VAL + 1):
        v = 0
        rem = x
        for i, p in enumerate(primes):
            if p * p > rem:
                break
            if rem % p == 0:
                exp = 0
                while rem % p == 0:
                    rem //= p
                    exp += 1
                if exp % 2 == 1:
                    v |= (1 << i)
        if rem > 1:
            if rem in prime_to_idx:
                v |= (1 << prime_to_idx[rem])
        vecs[x] = v
    
    target = vecs[N]
    
    # Check 1-element solutions
    for x in range(N + 1, MAX_VAL + 1):
        if vecs[x] == target:
            s = math.isqrt(x * N)
            if s * s == x * N:
                return [x]
    
    # Check 2-element solutions
    vec_to_nums = {}
    for x in range(N + 1, MAX_VAL + 1):
        vec_to_nums.setdefault(vecs[x], []).append(x)
    
    best_2max = MAX_VAL + 1
    best_2pair = None
    for v, nums in vec_to_nums.items():
        needed = v ^ target
        if needed in vec_to_nums:
            for a in nums:
                for b in vec_to_nums[needed]:
                    if a != b and max(a, b) < best_2max:
                        best_2max = max(a, b)
                        best_2pair = (a, b)
    
    if best_2pair is not None:
        a, b = best_2pair
        s = math.isqrt(a * b * N)
        if s * s == a * b * N:
            return [a, b]
    
    # Check 3-element solutions
    best_3max = MAX_VAL + 1
    best_3triple = None
    # For 3 elements: v(a)^v(b)^v(c) = target, i.e., v(c) = v(a)^v(b)^target
    # Iterate over pairs (a,b) and look up c
    all_nums = list(range(N + 1, MAX_VAL + 1))
    # To keep this fast, limit the search
    for idx_a in range(len(all_nums)):
        if time.time() > deadline - 10:
            break
        a = all_nums[idx_a]
        va = vecs[a]
        for idx_b in range(idx_a + 1, len(all_nums)):
            b = all_nums[idx_b]
            if max(a, b) >= best_3max:
                break
            vb = vecs[b]
            needed_c = va ^ vb ^ target
            if needed_c in vec_to_nums:
                for c in vec_to_nums[needed_c]:
                    if c != a and c != b:
                        m3 = max(a, b, c)
                        if m3 < best_3max:
                            best_3max = m3
                            best_3triple = (a, b, c)
    
    if best_3triple is not None:
        a, b, c = best_3triple
        s = math.isqrt(a * b * c * N)
        if s * s == a * b * c * N:
            return [a, b, c]
    
    # Incremental basis approach for larger solutions
    basis = [0] * (D + 1)
    origin = [set() for _ in range(D + 1)]
    null_space = []
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        v = vecs[M]
        
        temp = v
        temp_origin = {M}
        
        for i in range(D - 1, -1, -1):
            if (temp >> i) & 1:
                if basis[i] == 0:
                    basis[i] = temp
                    origin[i] = temp_origin
                    break
                else:
                    temp ^= basis[i]
                    temp_origin ^= origin[i]
        
        if temp == 0:
            null_space.append(temp_origin)
        
        # Check if target is in the span
        temp_t = target
        indices = set()
        in_span = True
        for i in range(D - 1, -1, -1):
            if (temp_t >> i) & 1:
                if basis[i] != 0:
                    temp_t ^= basis[i]
                    indices ^= origin[i]
                else:
                    in_span = False
                    break
        
        if in_span and temp_t == 0:
            sol = indices
            if M - N <= 100:
                prod = 1
                for x in sol:
                    prod *= x
                total = prod * N
                s = math.isqrt(total)
                if s * s == total:
                    return list(sol)
            else:
                if len(sol) > 100:
                    for _ in range(500):
                        changed = False
                        for ns in null_space:
                            if not ns:
                                continue
                            inter = len(ns & sol)
                            if 2 * inter > len(ns):
                                new_sol = sol ^ ns
                                if len(new_sol) < len(sol):
                                    sol = new_sol
                                    changed = True
                                    break
                        if not changed:
                            break
                if 1 <= len(sol) <= 100:
                    prod = 1
                    for x in sol:
                        prod *= x
                    total = prod * N
                    s = math.isqrt(total)
                    if s * s == total:
                        return list(sol)
    
    # Fallback
    return [840]
