import time
import numpy as np
from math import gcd, isqrt
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit

    MAX_N = 256

    # 1. Pre-compute compatibility matrix
    # compat[i][j] is True if gcd(i, j) == 1
    # Note: 1 is coprime to all.
    compat = np.ones((MAX_N + 1, MAX_N + 1), dtype=bool)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1):
            if gcd(i, j) != 1:
                compat[i, j] = False
                compat[j, i] = False

    def is_valid_set(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if not compat[S_list[i], S_list[j]]:
                    return False
        return True

    # 2. Greedy Initialization
    S = set()
    # Sort by value descending
    for n in range(MAX_N, 0, -1):
        # Check if n is coprime to all in S
        if all(compat[n, x] for x in S):
            S.add(n)
    
    best_set = set(S)
    best_sum = sum(S)
    
    # 3. Local Search (Hill Climbing)
    # Try to add or swap
    improved = True
    while improved and time.time() < deadline - 40: # Leave 40s for ILP
        improved = False
        S_list = list(best_set)
        
        # Try to add any missing number
        for b in range(MAX_N, 0, -1):
            if b in best_set:
                continue
            if all(compat[b, x] for x in best_set):
                best_set.add(b)
                best_sum += b
                improved = True
        
        # Try 1-for-1 swaps
        # We want to replace small a with large b
        for a in sorted(best_set):
            if a == 1:
                continue
            for b in range(MAX_N, a, -1):
                if b in best_set:
                    continue
                # Check if b is coprime to best_set \ {a}
                valid = True
                for x in best_set:
                    if x == a:
                        continue
                    if not compat[b, x]:
                        valid = False
                        break
                if valid:
                    best_set.remove(a)
                    best_set.add(b)
                    best_sum = best_sum - a + b
                    improved = True
                    break # Restart scan after a swap? Or continue? 
                    # Continuing might find more swaps, but S changed.
                    # Breaking ensures we re-evaluate the set state cleanly.
                    # However, breaking the inner loop only. We should probably break outer too or restart.
                    # Let's break inner and let the outer loop continue, but S is modified.
                    # The 'a' in outer loop is no longer in S. 
                    # This is safe enough for a heuristic.

    # 4. Simulated Annealing (if time permits)
    # This helps escape local optima
    if time.time() < deadline - 40:
        current_set = set(best_set)
        current_sum = best_sum
        T = 100.0
        cooling = 0.999
        min_T = 1.0
        
        while T > min_T and time.time() < deadline - 40:
            # Pick a random element to remove
            if not current_set:
                break
            a = random.choice(list(current_set))
            # Pick a random element to add
            candidates = [x for x in range(1, MAX_N + 1) if x not in current_set]
            if not candidates:
                break
            b = random.choice(candidates)
            
            # Check validity of swap
            valid = True
            for x in current_set:
                if x == a:
                    continue
                if not compat[b, x]:
                    valid = False
                    break
            
            if valid:
                new_sum = current_sum - a + b
                delta = new_sum - current_sum
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_set.remove(a)
                    current_set.add(b)
                    current_sum = new_sum
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = set(current_set)
            T *= cooling

    # 5. ILP Approach
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix

        # Build constraint matrix
        # Rows: Primes <= 256
        # Cols: Numbers 1..256
        primes = [i for i in range(2, MAX_N + 1) if all(i % j != 0 for j in range(2, isqrt(i) + 1))]
        prime_to_idx = {p: i for i, p in enumerate(primes)}
        
        rows = []
        cols = []
        data = []
        
        for n in range(1, MAX_N + 1):
            for p in primes:
                if n % p == 0:
                    idx = prime_to_idx[p]
                    rows.append(idx)
                    cols.append(n - 1)
                    data.append(1.0)
                    
        A = csr_matrix((data, (rows, cols)), shape=(len(primes), MAX_N))
        c = -np.arange(1, MAX_N + 1, dtype=float)
        constraints = LinearConstraint(A, lb=np.zeros(len(primes)), ub=np.ones(len(primes)))
        integrality = np.ones(MAX_N)
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))
        
        # Time limit for solver
        time_left = deadline - time.time() - 5
        if time_left > 2:
            options = {'time_limit': time_left}
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
            
            if res.status == 0 and res.x is not None:
                x = res.x
                selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]
                
                # Verify
                if len(selected) > 0:
                    valid = True
                    sel_set = set(selected)
                    for i in range(len(selected)):
                        for j in range(i + 1, len(selected)):
                            if not compat[selected[i], selected[j]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        ilp_sum = sum(selected)
                        if ilp_sum > best_sum:
                            best_sum = ilp_sum
                            best_set = set(selected)
    except Exception:
        pass

    # Final check and return
    # Ensure 1 is included if possible (it always is, gcd(1,x)=1)
    if 1 not in best_set:
        best_set.add(1)
        best_sum += 1

    return [int(x) for x in sorted(best_set)]
