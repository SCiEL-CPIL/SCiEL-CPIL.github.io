import math
import time
import random
import heapq
from itertools import product

MAX_N = 10**18
DEADLINE = 138.0  # 12 seconds before the 150s kill

def generate_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = generate_primes(5000)
PRIME_SET = set(PRIMES)

def get_divisors_sorted(primes_list, exponents_list):
    """
    Generate divisors in sorted order using a priority queue (multi-way merge of geometric progressions).
    primes_list: list of primes
    exponents_list: list of exponents corresponding to primes_list
    """
    if not primes_list:
        return [1]
    
    # We'll use a heap-based approach
    # Each element in the heap is (divisor, index, state)
    # state is a tuple of current exponents for each prime
    # We start from 1 and generate next divisors
    
    # For efficiency with many divisors, we use the standard "ugly numbers" generalization
    # But we need to avoid duplicates
    
    # Alternative: generate all divisors and sort (fast for moderate count)
    # For n up to 10^18, max divisors is around 100k-200k, which is manageable
    
    divs = [1]
    for p, e in zip(primes_list, exponents_list):
        if e == 0:
            continue
        powers = [1]
        for _ in range(e):
            powers.append(powers[-1] * p)
        new_divs = []
        for d in divs:
            for pow_p in powers:
                new_divs.append(d * pow_p)
        divs = new_divs
    
    divs.sort()
    return divs

def score_from_divs(divs):
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def evaluate_structure(primes_list, exponents_list):
    """
    Evaluate the score for a given prime exponent structure.
    Returns (score, n, num_divisors)
    """
    n = 1
    num_divs = 1
    for p, e in zip(primes_list, exponents_list):
        if e > 0:
            n *= (p ** e)
            if n > MAX_N:
                return -1, n, 0
            num_divs *= (e + 1)
    
    if n < 2:
        return -1, n, 0
    
    if num_divs > 500000:
        # Too many divisors to enumerate efficiently
        return -1, n, num_divs
    
    divs = get_divisors_sorted(primes_list, exponents_list)
    s = score_from_divs(divs)
    return s, n, len(divs)

def construct():
    start_time = time.time()
    
    # Strategy: 
    # 1. Start with squarefree (all exponents 1) for a good baseline
    # 2. Try mixed exponent structures
    # 3. Use local search to optimize
    
    # Initial best: squarefree with first 15 primes
    init_primes = PRIMES[:15]
    init_exps = [1] * 15
    best_score, best_n, _ = evaluate_structure(init_primes, init_exps)
    best_primes = init_primes.copy()
    best_exps = init_exps.copy()
    
    # Local search with diverse moves
    # Moves:
    # 1. Change exponent of a prime (increase/decrease)
    # 2. Replace a prime with another
    # 3. Add a new prime
    # 4. Remove a prime
    # 5. Simulated annealing-like exploration
    
    improved = True
    iteration = 0
    temperature = 10.0  # For simulated annealing-like behavior
    
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        iteration += 1
        temperature *= 0.95  # Cooling
        
        current_primes = best_primes.copy()
        current_exps = best_exps.copy()
        current_score = best_score
        
        current_n = 1
        for p, e in zip(current_primes, current_exps):
            if e > 0:
                current_n *= (p ** e)
        
        # Generate candidate moves
        candidates = []
        
        # 1. Change exponents
        for i, (p, e) in enumerate(zip(current_primes, current_exps)):
            # Increase exponent
            if e > 0 and current_n * p <= MAX_N:
                new_exps = current_exps.copy()
                new_exps[i] = e + 1
                candidates.append(('exp_inc', i, p, new_exps))
            
            # Decrease exponent
            if e > 1:
                new_exps = current_exps.copy()
                new_exps[i] = e - 1
                candidates.append(('exp_dec', i, p, new_exps))
            elif e == 1:
                # Remove prime
                new_primes = [p for j, p in enumerate(current_primes) if j != i]
                new_exps = [e for j, e in enumerate(current_exps) if j != i]
                candidates.append(('remove', i, p, (new_primes, new_exps)))
        
        # 2. Replace a prime with another
        used_primes = set(p for p, e in zip(current_primes, current_exps) if e > 0)
        for i, p in enumerate(current_primes):
            if current_exps[i] == 0:
                continue
            max_q = MAX_N * p // current_n
            valid_qs = [q for q in PRIMES if q <= max_q and q not in used_primes]
            # Limit to nearby primes for efficiency
            if valid_qs:
                # Take a sample or all if small
                if len(valid_qs) > 20:
                    sampled = random.sample(valid_qs, 20)
                else:
                    sampled = valid_qs
                for q in sampled:
                    new_primes = current_primes.copy()
                    new_primes[i] = q
                    candidates.append(('swap', i, q, (new_primes, current_exps.copy())))
        
        # 3. Add a new prime
        max_q = MAX_N // current_n
        valid_qs = [q for q in PRIMES if q <= max_q and q not in used_primes]
        if valid_qs:
            if len(valid_qs) > 15:
                sampled = random.sample(valid_qs, 15)
            else:
                sampled = valid_qs
            for q in sampled:
                new_primes = current_primes + [q]
                new_exps = current_exps + [1]
                candidates.append(('add', len(current_primes), q, (new_primes, new_exps)))
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        # Evaluate candidates with acceptance criteria
        for move_type, idx, prime, new_struct in candidates:
            if time.time() - start_time > DEADLINE:
                break
            
            if move_type in ('exp_inc', 'exp_dec', 'swap'):
                new_primes, new_exps = current_primes, new_struct if isinstance(new_struct, list) and len(new_struct) == len(current_primes) else (current_primes, new_struct)
                # Handle different move types
                if move_type == 'exp_inc' or move_type == 'exp_dec':
                    eval_primes = current_primes
                    eval_exps = new_struct
                else:  # swap
                    eval_primes, eval_exps = new_struct
            else:  # add, remove
                eval_primes, eval_exps = new_struct
            
            s, n, num_divs = evaluate_structure(eval_primes, eval_exps)
            if s < 0:
                continue
            
            # Acceptance: always accept if better, sometimes accept if worse (annealing)
            if s > current_score:
                best_score = s
                best_n = n
                best_primes = eval_primes
                best_exps = eval_exps
                improved = True
                break
            elif s == current_score and random.random() < 0.1:
                # Accept equal with small probability to escape plateaus
                current_primes = eval_primes
                current_exps = eval_exps
            elif temperature > 0.1 and random.random() < math.exp((s - current_score) / temperature):
                # Annealing acceptance
                current_primes = eval_primes
                current_exps = eval_exps
    
    return best_n
