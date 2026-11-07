import math
import random

def construct():
    max_val = 10210
    min_val = 211
    
    # 1. Sieve of Eratosthenes for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_to_index = {p: i for i, p in enumerate(primes)}
    d = len(primes)
    
    # 2. Compute parity vector for 210
    # 210 = 2 * 3 * 5 * 7
    t = 0
    for p in [2, 3, 5, 7]:
        t |= (1 << prime_to_index[p])
        
    # 3. Compute parity vectors for integers 211 to max_val
    # vectors[i] corresponds to integer 211 + i
    vectors = [0] * (max_val - 210)
    for i in range(max_val - 210):
        x = 211 + i
        v = 0
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                count = 0
                while n % p == 0:
                    n //= p
                    count += 1
                if count % 2 == 1:
                    v |= (1 << prime_to_index[p])
        if n > 1:
            if n in prime_to_index:
                v |= (1 << prime_to_index[n])
        vectors[i] = v

    # 4. Search for minimal K
    # We iterate K from 1 to max_val - 210
    # For each K, we check if there is a solution of weight <= 100
    
    # Pre-allocate some structures if needed, but rebuilding is fast enough for small K
    
    for K in range(1, max_val - 210 + 1):
        # Build rows for Gaussian Elimination
        # Row j corresponds to prime j.
        # Bit i of row j is 1 if prime j is in vectors[i].
        # Bit K of row j is 1 if prime j is in t.
        
        rows = []
        for j in range(d):
            row = 0
            for i in range(K):
                if (vectors[i] >> j) & 1:
                    row |= (1 << i)
            if (t >> j) & 1:
                row |= (1 << K)
            rows.append(row)
        
        # Gaussian Elimination to RREF
        # We want to find a particular solution and null space basis.
        pivot_row = 0
        pivot_cols = []
        
        for c in range(K):
            # Find pivot
            found = False
            for r in range(pivot_row, d):
                if (rows[r] >> c) & 1:
                    rows[r], rows[pivot_row] = rows[pivot_row], rows[r]
                    found = True
                    break
            if not found:
                continue
            
            # Eliminate other rows
            for r in range(d):
                if r != pivot_row and ((rows[r] >> c) & 1):
                    rows[r] ^= rows[pivot_row]
                    
            pivot_cols.append(c)
            pivot_row += 1
            if pivot_row >= d:
                break
        
        # Check consistency
        # If there is a row with 0 in all variable columns but 1 in RHS, it's inconsistent
        consistent = True
        for r in range(len(pivot_cols), d):
            # Check if row has any variable bits set
            var_bits = rows[r] & ((1 << K) - 1)
            if var_bits == 0:
                if (rows[r] >> K) & 1:
                    consistent = False
                    break
        
        if not consistent:
            continue
        
        # Extract particular solution (free vars = 0)
        c_p = 0
        for r in range(len(pivot_cols)):
            c = pivot_cols[r]
            if (rows[r] >> K) & 1:
                c_p |= (1 << c)
        
        # Build null space basis
        pivot_set = set(pivot_cols)
        free_cols = [c for c in range(K) if c not in pivot_set]
        
        null_basis = []
        for f in free_cols:
            n = 1 << f
            for r in range(len(pivot_cols)):
                c = pivot_cols[r]
                if (rows[r] >> f) & 1:
                    n |= (1 << c)
            null_basis.append(n)
        
        # Minimize weight of c_p using null_basis
        current = c_p
        cur_weight = bin(current).count('1')
        
        # If already <= 100, we can return immediately (though we might find a smaller weight, 
        # the score depends only on max value, which is fixed by K)
        # However, we must ensure the solution is valid.
        # The problem asks to minimize max(selected). So the first K that yields a valid solution is the answer.
        
        if cur_weight > 100:
            # Try to reduce weight
            # Greedy single flip
            improved = True
            while improved:
                improved = False
                for b in null_basis:
                    new_c = current ^ b
                    new_w = bin(new_c).count('1')
                    if new_w < cur_weight:
                        current = new_c
                        cur_weight = new_w
                        improved = True
            
            # If still > 100, try random restarts
            if cur_weight > 100:
                rng = random.Random(K) # Seed with K for reproducibility
                best_c = current
                best_w = cur_weight
                for _ in range(20):
                    # Random combination of null basis vectors
                    delta = 0
                    for b in null_basis:
                        if rng.random() < 0.5:
                            delta ^= b
                    new_c = c_p ^ delta
                    new_w = bin(new_c).count('1')
                    if new_w < best_w:
                        best_c = new_c
                        best_w = new_w
                    
                    # Greedy polish on best_c
                    cur = best_c
                    cw = best_w
                    imp = True
                    while imp:
                        imp = False
                        for b in null_basis:
                            nc = cur ^ b
                            nw = bin(nc).count('1')
                            if nw < cw:
                                cur = nc
                                cw = nw
                                imp = True
                    if cw < best_w:
                        best_c = cur
                        best_w = cw
                
                if best_w <= 100:
                    current = best_c
                    cur_weight = best_w
            
            if cur_weight > 100:
                # No solution with weight <= 100 for this K
                continue
        
        # Solution found with weight <= 100
        selected = []
        for i in range(K):
            if (current >> i) & 1:
                selected.append(211 + i)
        
        # Verify
        # prod = 210
        # for x in selected:
        #     prod *= x
        # if math.isqrt(prod)**2 != prod:
        #     raise Exception("Verification failed")
        
        return selected

    # Fallback: should not reach here if a solution exists
    # Return empty list or raise error?
    # The problem guarantees a solution exists? 
    # "1..100 distinct integers... Their product... must be a perfect square."
    # If no solution found, return empty list (which is invalid) or best effort.
    return []

if __name__ == "__main__":
    sol = construct()
    if sol:
        print(f"Solution size: {len(sol)}")
        print(f"Max element: {max(sol)}")
        prod = 210
        for x in sol:
            prod *= x
        sqrt_prod = math.isqrt(prod)
        is_square = (sqrt_prod * sqrt_prod == prod)
        print(f"Is square: {is_square}")
        print(f"Raw value: {max(sol) - 210}")
    else:
        print("No solution found")
