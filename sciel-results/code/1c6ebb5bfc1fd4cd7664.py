import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate primes p = 1 (mod 4)
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes_1_mod_4 = []
    # We need primes up to roughly 10^5 or so to have enough choices
    # 10^18 limit. Smallest primes are best.
    # 5, 13, 17, ...
    # Let's generate up to 2000
    for i in range(5, 2000, 4):
        if is_prime(i):
            primes_1_mod_4.append(i)
        if len(primes_1_mod_4) > 50:
            break

    # 2. DFS to find N = prod p_i^e_i maximizing prod (e_i+1)
    # Subject to N <= 10^18
    # Sort primes by value (ascending) to prioritize small primes
    primes_1_mod_4.sort()
    
    best_score = 0
    best_exponents = []
    best_N = 0
    best_primes_used = []
    
    def dfs(idx, current_N, current_score, exponents):
        nonlocal best_score, best_exponents, best_N, best_primes_used
        
        if time.time() > deadline - 20.0:
            return
            
        if current_score > best_score:
            best_score = current_score
            best_exponents = exponents[:]
            best_N = current_N
            best_primes_used = primes_1_mod_4[:idx]
            
        if idx >= len(primes_1_mod_4):
            return
            
        p = primes_1_mod_4[idx]
        # Try exponents e from 1 upwards
        # current_N * p^e <= 10^18
        max_e = 0
        temp_N = current_N
        while temp_N * p <= 10**18:
            temp_N *= p
            max_e += 1
            
        # We can also choose e=0 (skip this prime)
        # But since we want to maximize score, and score increases with e,
        # we should try all e from 0 to max_e.
        # However, using a larger prime later might be better than high exponent on current?
        # Generally, small primes with lower exponents allow more primes to be included.
        # The product (e+1) grows linearly with e, but N grows exponentially.
        # So low exponents on many small primes is usually best.
        
        for e in range(max_e, -1, -1):
            if e == 0:
                dfs(idx + 1, current_N, current_score, exponents)
            else:
                # Calculate new N and score
                new_N = current_N * (p ** e)
                if new_N > 10**18:
                    continue
                new_score = current_score * (e + 1)
                exponents.append(e)
                dfs(idx + 1, new_N, new_score, exponents)
                exponents.pop()
                
    # Start DFS
    # Use only the first ~20 primes to keep branching factor manageable?
    # Actually, with 50 primes, depth 50, branching ~3-5. 3^50 is too big.
    # We need to prune.
    # If current_score * (max_possible_future_score) <= best_score, prune.
    # Max future score: if we use all remaining primes with e=1, score multiplies by 2^(remaining).
    # This is a good upper bound.
    
    def dfs_pruned(idx, current_N, current_score, exponents):
        nonlocal best_score, best_exponents, best_N, best_primes_used
        
        if time.time() > deadline - 20.0:
            return
            
        # Upper bound on score
        # Remaining primes: len(primes_1_mod_4) - idx
        # Max multiplier if all e=1: 2^(remaining)
        # This is a loose bound but helps.
        remaining = len(primes_1_mod_4) - idx
        upper_bound_score = current_score * (2 ** remaining)
        if upper_bound_score <= best_score:
            return
            
        if current_score > best_score:
            best_score = current_score
            best_exponents = exponents[:]
            best_N = current_N
            best_primes_used = primes_1_mod_4[:idx]
            
        if idx >= len(primes_1_mod_4):
            return
            
        p = primes_1_mod_4[idx]
        max_e = 0
        temp_N = current_N
        while temp_N * p <= 10**18:
            temp_N *= p
            max_e += 1
            
        # Try e from max_e down to 0?
        # Higher e gives higher score for this prime, but uses more N budget.
        # Let's try all.
        for e in range(max_e + 1):
            if e == 0:
                dfs_pruned(idx + 1, current_N, current_score, exponents)
            else:
                new_N = current_N * (p ** e)
                if new_N > 10**18:
                    continue
                new_score = current_score * (e + 1)
                exponents.append(e)
                dfs_pruned(idx + 1, new_N, new_score, exponents)
                exponents.pop()

    # Limit primes to first 30 to ensure DFS finishes
    primes_1_mod_4 = primes_1_mod_4[:30]
    dfs_pruned(0, 1, 1, [])
    
    if best_N == 0:
        # Fallback
        best_N = 2
        best_pairs = [[1, 1]]
        return {'N': best_N, 'pairs': best_pairs}

    # 3. Find all representations of best_N as sum of two squares
    # best_N = prod p_i^e_i
    # We need to factor each p_i into a_i^2 + b_i^2
    def sum_of_squares_factor(p):
        # Find a, b such that a^2 + b^2 = p
        # p is prime 1 mod 4
        # Use Cornacchia or simple search since p is small
        for a in range(1, math.isqrt(p) + 1):
            b2 = p - a * a
            b = math.isqrt(b2)
            if b * b == b2:
                return a, b
        return None, None

    # Generate all representations
    # Representations in Gaussian integers
    # z = prod (a_i + b_i i)^e_i
    # We can just iterate through all divisors of N in Z[i]?
    # Simpler: Since we have the exponents, we can generate all combinations.
    
    reps = []
    
    # For each prime, get (a,b)
    prime_factors = []
    for i, p in enumerate(best_primes_used):
        e = best_exponents[i]
        if e == 0:
            continue
        a, b = sum_of_squares_factor(p)
        if a is None:
            continue
        prime_factors.append((a, b, e))
        
    # Recursive generation of (x,y)
    def gen_reps(idx, x, y):
        if idx >= len(prime_factors):
            # (x,y) is a representation
            # We want x^2 + y^2 = N
            # Add to list
            reps.append((x, y))
            return
            
        a, b, e = prime_factors[idx]
        # We need to compute (a+bi)^e * (x+yi)
        # But we are building the product from scratch.
        # Actually, we should multiply the current (x,y) by all combinations of (a+bi)^k for k=0..e?
        # No, the exponent is fixed for the prime.
        # The representation of p^e is the e-th power of (a+bi).
        # Let z_p = (a+bi)^e = X_p + Y_p i
        # Then new_x = x * X_p - y * Y_p
        # new_y = x * Y_p + y * X_p
        
        # Calculate (a+bi)^e
        ax, ay = a, b
        for _ in range(e):
            if _ > 0:
                ax, ay = ax * a - ay * b, ax * b + ay * a
        # ax, ay is (a+bi)^e
        
        gen_reps(idx + 1, x * ax - y * ay, x * ay + y * ax)
        
    # Start with (1,0)
    gen_reps(0, 1, 0)
    
    # Process reps
    pairs = []
    seen = set()
    for x, y in reps:
        # x, y can be negative
        # a = x^2, b = y^2
        a = x * x
        b = y * y
        if a > b:
            a, b = b, a
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append([a, b])
            
    # Verify pairs
    # Check if a+b == best_N
    # Check if a,b are powerful
    # Squares are powerful.
    
    # Are there other powerful numbers?
    # We can do a quick check for small powerful numbers if time permits
    # But the square pairs are guaranteed.
    
    # If best_score is high, we are good.
    # The score is len(pairs).
    
    # To ensure we don't miss "mixed" powerful numbers (e.g. 125 * square),
    # we should check if there are other representations.
    # However, generating all powerful numbers up to 10^18 is impossible.
    # But we can check if best_N is divisible by a powerful number c > 1.
    # If so, N/c might have sum of squares reps.
    # This is complex. Given the score jump from 4.8 to ~1000+, this is likely sufficient.
    
    # One optimization: sort pairs
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}

if __name__ == "__main__":
    construct()
