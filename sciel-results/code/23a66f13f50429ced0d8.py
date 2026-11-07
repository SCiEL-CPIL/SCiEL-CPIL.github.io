import math
import time

def construct():
    start = time.time()
    deadline = 85.0
    
    max_val = 10210
    
    # Sieve primes up to max_val
    is_prime = bytearray([1]) * (max_val + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = 0
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    pidx = {p: i for i, p in enumerate(primes)}
    d = len(primes)
    
    # Parity vector of 210 = 2*3*5*7
    t = 0
    for p in (2, 3, 5, 7):
        t |= (1 << pidx[p])
    
    # Precompute parity vectors for all numbers 211..max_val
    N = max_val - 210  # 10000
    vectors = [0] * N
    for i in range(N):
        x = 211 + i
        v = 0
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                cnt = 0
                while n % p == 0:
                    n //= p
                    cnt += 1
                if cnt & 1:
                    v |= (1 << pidx[p])
        if n > 1:
            v |= (1 << pidx[n])
        vectors[i] = v
    
    # Binary search on the answer: find minimum M such that
    # t is in the span of {vectors[0], ..., vectors[M-211]}
    # and the solution has weight <= 100
    
    def can_solve(K):
        """Check if t is in span of vectors[0..K-1] with weight <= 100 solution."""
        if K <= 0:
            return False
        # Build basis incrementally
        basis = [0] * (d + 1)
        rank = 0
        for i in range(K):
            v = vectors[i]
            if v == 0:
                continue
            r = v
            while r:
                msb = r.bit_length() - 1
                if basis[msb]:
                    r ^= basis[msb]
                else:
                    basis[msb] = r
                    rank += 1
                    break
        # Check if t is in span
        chk = t
        while chk:
            msb = chk.bit_length() - 1
            if basis[msb]:
                chk ^= basis[msb]
            else:
                return False
        return True
    
    # Find minimum K such that t is in span
    lo, hi = 1, N
    min_K = N
    while lo < hi:
        mid = (lo + hi) // 2
        if can_solve(mid):
            hi = mid
        else:
            lo = mid + 1
    min_K = lo
    
    # Now find the actual solution with minimum max index
    # Use Gaussian elimination on the first min_K vectors
    K = min_K
    
    # Build augmented matrix: d rows, K+1 columns
    rows = []
    for j in range(d):
        row = 0
        for i in range(K):
            if (vectors[i] >> j) & 1:
                row |= (1 << i)
        if (t >> j) & 1:
            row |= (1 << K)
        rows.append(row)
    
    # Gaussian elimination
    pivot_row = 0
    pivot_cols = []
    for c in range(K):
        found = -1
        for r in range(pivot_row, d):
            if (rows[r] >> c) & 1:
                found = r
                break
        if found == -1:
            continue
        rows[found], rows[pivot_row] = rows[pivot_row], rows[found]
        for r in range(d):
            if r != pivot_row and ((rows[r] >> c) & 1):
                rows[r] ^= rows[pivot_row]
        pivot_cols.append(c)
        pivot_row += 1
        if pivot_row >= d:
            break
    
    # Particular solution
    c_p = 0
    for r, c in enumerate(pivot_cols):
        if (rows[r] >> K) & 1:
            c_p |= (1 << c)
    
    # Null space basis
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(K) if c not in pivot_set]
    null_basis = []
    for f in free_cols:
        n = 1 << f
        for r, c in enumerate(pivot_cols):
            if (rows[r] >> f) & 1:
                n |= (1 << c)
        null_basis.append(n)
    
    # Greedy minimize max index
    current = c_p
    best_c = current
    best_max = current.bit_length() - 1 if current else -1
    
    # Single flip
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for b in null_basis:
            nc = current ^ b
            if nc == 0:
                continue
            nw = bin(nc).count('1')
            if nw > 100:
                continue
            nmax = nc.bit_length() - 1
            if nmax < best_max:
                current = nc
                best_c = nc
                best_max = nmax
                improved = True
    
    # Two flips
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            nb = len(null_basis)
            for i in range(nb):
                for j in range(i + 1, nb):
                    nc = current ^ null_basis[i] ^ null_basis[j]
                    if nc == 0:
                        continue
                    nw = bin(nc).count('1')
                    if nw > 100:
                        continue
                    nmax = nc.bit_length() - 1
                    if nmax < best_max:
                        current = nc
                        best_c = nc
                        best_max = nmax
                        improved = True
                        break
                if improved:
                    break
    
    # Build result
    selected = []
    for i in range(K):
        if (best_c >> i) & 1:
            selected.append(211 + i)
    
    if not selected:
        for i in range(K):
            if (c_p >> i) & 1:
                selected.append(211 + i)
    
    # Verify
    total_parity = t
    for x in selected:
        total_parity ^= vectors[x - 211]
    if total_parity != 0:
        selected = []
        for i in range(K):
            if (c_p >> i) & 1:
                selected.append(211 + i)
    
    return [int(x) for x in selected]

if __name__ == "__main__":
    sol = construct()
    print(f"Solution size: {len(sol)}")
    print(f"Max element: {max(sol) if sol else 0}")
    if sol:
        prod = 210
        for x in sol:
            prod *= x
        sqrt_prod = math.isqrt(prod)
        is_square = (sqrt_prod * sqrt_prod == prod)
        print(f"Is square: {is_square}")
        print(f"Raw value: {max(sol) - 210}")
