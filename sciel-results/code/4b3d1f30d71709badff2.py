import math
import time
import random

MAX_N = 10**18
DEADLINE = 140.0

def generate_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = generate_primes(1000)
PRIME_SET = set(PRIMES)

def score_from_divs(divs):
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def evaluate(exponents):
    n = 1
    for p, e in exponents.items():
        if e > 0:
            n *= (p ** e)
            if n > MAX_N:
                return -1, n
    
    if n < 2:
        return -1, n
        
    # Generate divisors
    divs = [1]
    for p, e in exponents.items():
        if e == 0:
            continue
        new_divs = []
        powers = [1]
        for _ in range(e):
            powers.append(powers[-1] * p)
        for d in divs:
            for pow_p in powers:
                new_divs.append(d * pow_p)
        divs = new_divs
    divs.sort()
    
    s = score_from_divs(divs)
    return s, n

def construct():
    start_time = time.time()
    
    # Initial best: First 15 primes, exponent 1
    init_exponents = {p: 1 for p in PRIMES[:15]}
    best_score, best_n = evaluate(init_exponents)
    best_exponents = init_exponents.copy()
    
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        current_exponents = best_exponents.copy()
        current_score = best_score
        
        current_n = 1
        for p, e in current_exponents.items():
            if e > 0:
                current_n *= (p ** e)
                
        candidates = []
        
        # 1. Increase exponent
        for p, e in current_exponents.items():
            if e > 0:
                if current_n * p <= MAX_N:
                    new_exp = current_exponents.copy()
                    new_exp[p] = e + 1
                    candidates.append(new_exp)
                    
        # 2. Decrease exponent
        for p, e in current_exponents.items():
            if e > 1:
                new_exp = current_exponents.copy()
                new_exp[p] = e - 1
                candidates.append(new_exp)
            elif e == 1:
                new_exp = current_exponents.copy()
                del new_exp[p]
                candidates.append(new_exp)
                
        used_primes = set(p for p, e in current_exponents.items() if e > 0)
        
        swap_candidates = []
        for p in used_primes:
            max_q = MAX_N * p // current_n
            valid_qs = [q for q in PRIMES if q <= max_q and q not in used_primes]
            if valid_qs:
                # Limit swaps to avoid explosion
                if len(valid_qs) > 20:
                    valid_qs = random.sample(valid_qs, 20)
                for q in valid_qs:
                    new_exp = current_exponents.copy()
                    del new_exp[p]
                    new_exp[q] = 1
                    swap_candidates.append(new_exp)
        
        # 4. Add new prime
        max_q = MAX_N // current_n
        valid_qs = [q for q in PRIMES if q <= max_q and q not in used_primes]
        if valid_qs:
            if len(valid_qs) > 10:
                valid_qs = random.sample(valid_qs, 10)
            for q in valid_qs:
                new_exp = current_exponents.copy()
                new_exp[q] = 1
                candidates.append(new_exp)

        candidates.extend(swap_candidates)
        
        random.shuffle(candidates)
        
        for new_exp in candidates:
            if time.time() - start_time > DEADLINE:
                break
                
            s, n = evaluate(new_exp)
            if s > best_score:
                best_score = s
                best_n = n
                best_exponents = new_exp
                improved = True
                break
                
    return best_n
