import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    num_primes = len(primes)
    
    def get_divisors_from_exp(exp_list):
        divs = [1]
        for i, e in enumerate(exp_list):
            if e == 0:
                continue
            p = primes[i]
            new_divs = []
            power = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * power)
                power *= p
            divs = new_divs
        return sorted(divs)
    
    def calc_score(exp_list):
        divs = get_divisors_from_exp(exp_list)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    def get_n_from_exp(exp_list):
        n = 1
        for i, e in enumerate(exp_list):
            if e > 0:
                n *= (primes[i] ** e)
        return n
    
    # Global best
    best_n = 2
    best_score = 0
    best_exp = [0] * num_primes
    
    def update_best(exp_list):
        nonlocal best_n, best_score, best_exp
        n = get_n_from_exp(exp_list)
        if n < 2 or n > 10**18:
            return
        score = calc_score(exp_list)
        if score > best_score:
            best_score = score
            best_n = n
            best_exp = exp_list[:]
    
    # Phase 1: DFS WITHOUT monotone constraint - allow any exponent pattern
    # We explore by choosing which primes to use and what exponents (no ordering constraint)
    
    def dfs_free(idx, current_prod, exp_list):
        nonlocal best_score
        if time.time() > deadline:
            return
        
        # Option: stop here (all remaining exponents are 0)
        if current_prod >= 2:
            # Quick upper bound: max possible score is tau - 1
            tau = 1
            for i in range(idx):
                tau *= (exp_list[i] + 1)
            if tau - 1 > best_score:
                update_best(exp_list[:idx] + [0] * (num_primes - idx))
        
        # Option: continue with more primes
        if idx >= num_primes:
            return
        
        p = primes[idx]
        
        # Try exponents from reasonable_max down to 1
        # Reasonable max: p^e <= 10^18 / current_prod
        prod_limit = 10**18 // current_prod if current_prod > 0 else 10**18
        max_e = 0
        pe = 1
        while pe * p <= prod_limit:
            pe *= p
            max_e += 1
        
        # Limit max_e to avoid too many divisors (performance)
        max_e = min(max_e, 12)
        
        for e in range(max_e, 0, -1):
            new_prod = current_prod * (p ** e)
            if new_prod > 10**18:
                continue
            exp_list[idx] = e
            dfs_free(idx + 1, new_prod, exp_list)
            exp_list[idx] = 0
    
    exp_list = [0] * num_primes
    dfs_free(0, 1, exp_list)
    
    # Phase 2: Local refinement with broader neighborhood
    def local_refine(exp_list, max_iterations=1000):
        nonlocal best_score, best_n, best_exp
        current_exp = exp_list[:]
        current_n = get_n_from_exp(current_exp)
        current_score = calc_score(current_exp)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try each prime index with various adjustments
            for idx in range(num_primes):
                # Try +1, +2, -1, -2
                for delta in [1, 2, -1, -2]:
                    new_exp = current_exp[:]
                    new_exp[idx] += delta
                    if new_exp[idx] < 0:
                        continue
                    new_n = get_n_from_exp(new_exp)
                    if new_n <= 10**18 and new_n >= 2:
                        new_score = calc_score(new_exp)
                        if new_score > current_score:
                            current_exp = new_exp
                            current_n = new_n
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
                                best_exp = current_exp[:]
                
                # Try swapping exponents between two primes
                for idx2 in range(idx + 1, num_primes):
                    if current_exp[idx] != current_exp[idx2]:
                        new_exp = current_exp[:]
                        new_exp[idx], new_exp[idx2] = new_exp[idx2], new_exp[idx]
                        new_n = get_n_from_exp(new_exp)
                        if new_n <= 10**18 and new_n >= 2:
                            new_score = calc_score(new_exp)
                            if new_score > current_score:
                                current_exp = new_exp
                                current_n = new_n
                                current_score = new_score
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_n = current_n
                                    best_exp = current_exp[:]
        
        return current_exp
    
    # Refine the best found so far
    if best_score > 0:
        refined = local_refine(best_exp, max_iterations=500)
    
    # Phase 3: Additional diverse candidate generation
    # Try all combinations of 2-3 primes with varying exponents
    for i in range(num_primes):
        if time.time() > deadline:
            break
        for j in range(i + 1, min(i + 10, num_primes)):
            if time.time() > deadline:
                break
            for e_i in range(1, 8):
                if time.time() > deadline:
                    break
                for e_j in range(1, 8):
                    if time.time() > deadline:
                        break
                    exp_test = [0] * num_primes
                    exp_test[i] = e_i
                    exp_test[j] = e_j
                    prod = (primes[i] ** e_i) * (primes[j] ** e_j)
                    # Add remaining primes with exponent 1
                    for k in range(num_primes):
                        if k == i or k == j:
                            continue
                        if prod * primes[k] > 10**18:
                            break
                        exp_test[k] = 1
                        prod *= primes[k]
                    if prod >= 2:
                        update_best(exp_test)
    
    # Phase 4: Random restarts with refinement
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        exp_test = [0] * num_primes
        # Random subset of primes
        k = random.randint(3, 12)
        chosen = random.sample(range(num_primes), k)
        for idx in chosen:
            exp_test[idx] = random.randint(1, 4)
        prod = get_n_from_exp(exp_test)
        if prod <= 10**18 and prod >= 2:
            update_best(exp_test)
            if best_score > 0:
                local_refine(best_exp, max_iterations=100)
    
    # Final refinement on the best
    if best_score > 0 and time.time() < deadline:
        refined = local_refine(best_exp, max_iterations=500)
    
    return best_n

if __name__ == "__main__":
    n = construct()
    print(f"Constructed: {n}")
