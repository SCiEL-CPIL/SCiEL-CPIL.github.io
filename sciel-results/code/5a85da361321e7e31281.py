import math
import time
import sys
from math import gcd
from itertools import product

def generate_divisors(primes, exponents):
    """Generate all divisors of n = prod(p_i^e_i)."""
    divs = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    return divs

def tau_perp(n, primes, exponents):
    """Count adjacent coprime pairs among sorted divisors of n."""
    divs = generate_divisors(primes, exponents)
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    # First 20 primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    # Find the largest primorial <= 10^18
    LIMIT = 10**18
    primorial = 1
    base_primes = []
    for p in small_primes:
        if primorial * p <= LIMIT:
            primorial *= p
            base_primes.append(p)
        else:
            break
    
    # Start with base exponents: 1 for each prime in base_primes, 0 for others
    k = len(base_primes)
    all_primes = small_primes[:max(k + 5, 20)]  # consider a few extra primes
    n_all = len(all_primes)
    
    # Initial exponents
    exponents = [0] * n_all
    for i in range(k):
        exponents[i] = 1
    
    def n_value(exps):
        v = 1
        for i, e in enumerate(exps):
            if e > 0:
                v *= all_primes[i] ** e
                if v > LIMIT:
                    return v
        return v
    
    def is_valid(exps):
        return n_value(exps) <= LIMIT
    
    # Evaluate initial
    if not is_valid(exponents):
        # Fallback
        return 614889782588491410
    
    best_exp = exponents[:]
    best_score = tau_perp(0, all_primes, exponents)
    
    # Local search: try adjusting exponents
    # We focus on the first few primes (2, 3, 5, 7, 11, 13, 17, 19)
    search_primes = min(8, n_all)
    
    # Multi-start: try different initial configurations
    configs = []
    
    # Config 1: primorial
    configs.append(exponents[:])
    
    # Config 2: primorial with 2^2
    e2 = exponents[:]
    e2[0] = 2
    if is_valid(e2):
        configs.append(e2)
    
    # Config 3: primorial with 2^3
    e3 = exponents[:]
    e3[0] = 3
    if is_valid(e3):
        configs.append(e3)
    
    # Config 4: primorial with 2^4
    e4 = exponents[:]
    e4[0] = 4
    if is_valid(e4):
        configs.append(e4)
    
    # Config 5: primorial with 3^2
    e5 = exponents[:]
    e5[1] = 2
    if is_valid(e5):
        configs.append(e5)
    
    # Config 6: primorial with 2^2, 3^2
    e6 = exponents[:]
    e6[0] = 2
    e6[1] = 2
    if is_valid(e6):
        configs.append(e6)
    
    # Config 7: primorial with 2^3, 3^2
    e7 = exponents[:]
    e7[0] = 3
    e7[1] = 2
    if is_valid(e7):
        configs.append(e7)
    
    # Config 8: primorial with 2^4, 3^2
    e8 = exponents[:]
    e8[0] = 4
    e8[1] = 2
    if is_valid(e8):
        configs.append(e8)
    
    # Config 9: primorial with 2^4, 3^3
    e9 = exponents[:]
    e9[0] = 4
    e9[1] = 3
    if is_valid(e9):
        configs.append(e9)
    
    # Config 10: primorial with 2^5, 3^2
    e10 = exponents[:]
    e10[0] = 5
    e10[1] = 2
    if is_valid(e10):
        configs.append(e10)
    
    # Config 11: primorial with 2^6, 3^2
    e11 = exponents[:]
    e11[0] = 6
    e11[1] = 2
    if is_valid(e11):
        configs.append(e11)
    
    # Config 12: primorial with 2^7, 3^2
    e12 = exponents[:]
    e12[0] = 7
    e12[1] = 2
    if is_valid(e12):
        configs.append(e12)
    
    # Config 13: primorial with 2^8, 3^2
    e13 = exponents[:]
    e13[0] = 8
    e13[1] = 2
    if is_valid(e13):
        configs.append(e13)
    
    # Config 14: primorial with 2^9, 3^2
    e14 = exponents[:]
    e14[0] = 9
    e14[1] = 2
    if is_valid(e14):
        configs.append(e14)
    
    # Config 15: primorial with 2^10, 3^2
    e15 = exponents[:]
    e15[0] = 10
    e15[1] = 2
    if is_valid(e15):
        configs.append(e15)
    
    # Config 16: primorial with 2^4, 3^2, 5^2
    e16 = exponents[:]
    e16[0] = 4
    e16[1] = 2
    e16[2] = 2
    if is_valid(e16):
        configs.append(e16)
    
    # Config 17: primorial with 2^4, 3^2, 5^3
    e17 = exponents[:]
    e17[0] = 4
    e17[1] = 2
    e17[2] = 3
    if is_valid(e17):
        configs.append(e17)
    
    # Config 18: primorial with 2^5, 3^3, 5^2
    e18 = exponents[:]
    e18[0] = 5
    e18[1] = 3
    e18[2] = 2
    if is_valid(e18):
        configs.append(e18)
    
    # Config 19: primorial with 2^6, 3^3, 5^2
    e19 = exponents[:]
    e19[0] = 6
    e19[1] = 3
    e19[2] = 2
    if is_valid(e19):
        configs.append(e19)
    
    # Config 20: primorial with 2^6, 3^3, 5^3
    e20 = exponents[:]
    e20[0] = 6
    e20[1] = 3
    e20[2] = 3
    if is_valid(e20):
        configs.append(e20)
    
    # For each config, do local search
    for cfg in configs:
        if time.time() > deadline:
            break
        if not is_valid(cfg):
            continue
        
        cur_exp = cfg[:]
        cur_score = tau_perp(0, all_primes, cur_exp)
        
        if cur_score > best_score:
            best_score = cur_score
            best_exp = cur_exp[:]
        
        # Local search: try incrementing each exponent
        improved = True
        max_iter = 50
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
            
            for i in range(search_primes):
                if time.time() > deadline:
                    break
                
                # Try incrementing
                new_exp = cur_exp[:]
                new_exp[i] += 1
                if is_valid(new_exp):
                    new_score = tau_perp(0, all_primes, new_exp)
                    if new_score > cur_score:
                        cur_score = new_score
                        cur_exp = new_exp
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_exp = cur_exp[:]
                
                # Try decrementing (if > 1)
                if cur_exp[i] > 1:
                    new_exp2 = cur_exp[:]
                    new_exp2[i] -= 1
                    if is_valid(new_exp2):
                        new_score2 = tau_perp(0, all_primes, new_exp2)
                        if new_score2 > cur_score:
                            cur_score = new_score2
                            cur_exp = new_exp2
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_exp = cur_exp[:]
                
                # Try setting to 1 (if currently > 1)
                if cur_exp[i] > 1:
                    new_exp3 = cur_exp[:]
                    new_exp3[i] = 1
                    if is_valid(new_exp3):
                        new_score3 = tau_perp(0, all_primes, new_exp3)
                        if new_score3 > cur_score:
                            cur_score = new_score3
                            cur_exp = new_exp3
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_exp = cur_exp[:]
                
                # Try setting to 0 (if currently > 0)
                if cur_exp[i] > 0:
                    new_exp4 = cur_exp[:]
                    new_exp4[i] = 0
                    if is_valid(new_exp4):
                        new_score4 = tau_perp(0, all_primes, new_exp4)
                        if new_score4 > cur_score:
                            cur_score = new_score4
                            cur_exp = new_exp4
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_exp = cur_exp[:]
    
    # Also try a broader search: random perturbations
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Random perturbation of best_exp
            cand = best_exp[:]
            # Randomly adjust 1-3 exponents
            num_changes = random.randint(1, 3)
            for _ in range(num_changes):
                i = random.randint(0, search_primes - 1)
                delta = random.randint(-2, 2)
                cand[i] = max(0, cand[i] + delta)
            
            if not is_valid(cand):
                continue
            
            score = tau_perp(0, all_primes, cand)
            if score > best_score:
                best_score = score
                best_exp = cand[:]
    
    # Compute final n
    result = 1
    for i, e in enumerate(best_exp):
        if e > 0:
            result *= all_primes[i] ** e
    
    if result < 2:
        result = 614889782588491410
    if result > LIMIT:
        result = 614889782588491410
    
    return result
