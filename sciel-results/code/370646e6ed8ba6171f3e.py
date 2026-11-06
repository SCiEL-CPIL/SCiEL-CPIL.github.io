import math
import time
import itertools
import random
from itertools import combinations
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    LIMIT = 10**18

    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divisors = [1]
        for p, e in factors:
            powers = [p ** k for k in range(e + 1)]
            new_divs = []
            for d in divisors:
                for pw in powers:
                    new_divs.append(d * pw)
            divisors = new_divs
        divisors.sort()
        return divisors

    def tau_perp(n, factors=None):
        """Compute the score: number of adjacent coprime pairs in sorted divisors."""
        if factors is None:
            factors = []
            temp = n
            p = 2
            while p * p <= temp:
                if temp % p == 0:
                    e = 0
                    while temp % p == 0:
                        temp //= p
                        e += 1
                    factors.append((p, e))
                p += 1 if p == 2 else 2
            if temp > 1:
                factors.append((temp, 1))
        
        divisors = get_divisors_from_factors(factors)
        score = 0
        gcd = math.gcd
        for i in range(len(divisors) - 1):
            if gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score, len(divisors)

    best_n = 1
    best_score = 0

    # Helper: evaluate a candidate given its factor list
    def try_candidate(factors):
        nonlocal best_n, best_score
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT:
            return
        score, _ = tau_perp(n, factors)
        if score > best_score:
            best_score = score
            best_n = n
        return score, n

    # Baseline: product of first 15 primes
    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_base = 1
    for p in primes_15:
        n_base *= p
    if n_base <= LIMIT:
        score_base, _ = tau_perp(n_base, [(p, 1) for p in primes_15])
        best_n = n_base
        best_score = score_base

    if time.time() - start_time > DEADLINE:
        return best_n

    # === GREEDY INTERVAL-MERGING HEURISTIC ===
    # Model: each prime p with exponent e creates "intervals" in log-space.
    # Adjacent coprime pairs occur at boundaries between divisors with disjoint supports.
    # We greedily select primes/exponents to maximize boundary density within budget.
    #
    # Key insight: for squarefree n with primes p1 < p2 < ... < pk,
    # the divisors sorted by value have coprime adjacencies where consecutive
    # divisors share no prime factor. The density of such pairs depends on
    # how "spread out" the primes are in log-space.
    #
    # Greedy strategy:
    # 1. Start with the smallest primes (maximizes count within budget)
    # 2. Try swapping in larger primes to improve log-spacing
    # 3. Try adding exponents to small primes to create more divisors
    # 4. Use a scoring heuristic to guide search

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    # Generate primes up to 300
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in range(101, 300):
        if is_prime(p):
            small_primes.append(p)

    # --- Phase 1: Explore 15-prime squarefree sets with log-spacing optimization ---
    # Strategy: Use multiple fixed sets and explore different combinations
    # to find better log-spacing configurations
    
    # First, try the baseline 15-prime set
    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_base = 1
    for p in primes_15:
        n_base *= p
    if n_base <= LIMIT:
        score_base, _ = tau_perp(n_base, [(p, 1) for p in primes_15])
        if score_base > best_score:
            best_score = score_base
            best_n = n_base

    if time.time() - start_time > DEADLINE:
        return best_n

    # Try multiple fixed-prefix strategies
    # Strategy A: Fix first 11, optimize last 4
    fixed_primes_A = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    fixed_prod_A = 1
    for p in fixed_primes_A:
        fixed_prod_A *= p

    candidates_A = [p for p in small_primes if p >= 37 and p <= 250]
    comb_count_A = 0
    for combo in combinations(candidates_A, 4):
        if time.time() - start_time > DEADLINE:
            break
        n = fixed_prod_A * combo[0] * combo[1] * combo[2] * combo[3]
        if n > LIMIT:
            continue
        factors = [(p, 1) for p in fixed_primes_A] + [(combo[0], 1), (combo[1], 1), (combo[2], 1), (combo[3], 1)]
        score, _ = tau_perp(n, factors)
        if score > best_score:
            best_score = score
            best_n = n
        comb_count_A += 1
        if comb_count_A % 2000 == 0:
            if time.time() - start_time > DEADLINE:
                break

    if time.time() - start_time > DEADLINE:
        return best_n

    # Strategy B: Fix first 12, optimize last 3
    fixed_primes_B = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    fixed_prod_B = 1
    for p in fixed_primes_B:
        fixed_prod_B *= p

    candidates_B = [p for p in small_primes if p >= 41 and p <= 200]
    comb_count_B = 0
    for combo in combinations(candidates_B, 3):
        if time.time() - start_time > DEADLINE:
            break
        n = fixed_prod_B * combo[0] * combo[1] * combo[2]
        if n > LIMIT:
            continue
        factors = [(p, 1) for p in fixed_primes_B] + [(combo[0], 1), (combo[1], 1), (combo[2], 1)]
        score, _ = tau_perp(n, factors)
        if score > best_score:
            best_score = score
            best_n = n
        comb_count_B += 1
        if comb_count_B % 2000 == 0:
            if time.time() - start_time > DEADLINE:
                break

    if time.time() - start_time > DEADLINE:
        return best_n

    # --- Phase 2: Try 14 primes with one exponent-2 (more divisors: 3*2^13 = 24576) ---
    # vs 15 squarefree: 2^15 = 32768 divisors
    # But the coprime density may differ.
    
    primes_14 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    n_14_base = 1
    for p in primes_14:
        n_14_base *= p
    
    # Try each prime with exponent 2
    for i, p in enumerate(primes_14):
        if time.time() - start_time > DEADLINE:
            break
        n_test = n_14_base * p  # multiply by p to make exponent 2
        if n_test > LIMIT:
            continue
        factors = [(pp, 2 if pp == p else 1) for pp in primes_14]
        score, _ = tau_perp(n_test, factors)
        if score > best_score:
            best_score = score
            best_n = n_test

    if time.time() - start_time > DEADLINE:
        return best_n

    # --- Phase 3: Try 13 primes with two exponents-2 (4*2^12 = 16384 divisors) ---
    # Or 13 primes with one exponent-3 (4*2^12 = 16384)
    primes_13 = primes_14[:13]
    n_13_base = 1
    for p in primes_13:
        n_13_base *= p

    # Try pairs of primes with exponent 2
    for i in range(len(primes_13)):
        for j in range(i+1, len(primes_13)):
            if time.time() - start_time > DEADLINE:
                break
            n_test = n_13_base * primes_13[i] * primes_13[j]
            if n_test > LIMIT:
                continue
            factors = [(pp, 2 if pp in (primes_13[i], primes_13[j]) else 1) for pp in primes_13]
            score, _ = tau_perp(n_test, factors)
            if score > best_score:
                best_score = score
                best_n = n_test

    if time.time() - start_time > DEADLINE:
        return best_n

    # --- Phase 4: Enhanced local search with multiple starting points ---
    # Try to find better 15-prime sets through iterative single-prime swaps
    
    # Start from the current best if it's a 15-prime squarefree number
    # Otherwise start from baseline
    def get_prime_set_from_n(n):
        """Extract prime factors from n (assumes squarefree)."""
        factors = []
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                factors.append(p)
                temp //= p
                while temp % p == 0:
                    temp //= p
            p += 1 if p == 2 else 2
        if temp > 1:
            factors.append(temp)
        return factors

    # Try local search from multiple starting points
    starting_points = [
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 53],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 53],
    ]
    
    for start_primes in starting_points:
        if time.time() - start_time > DEADLINE:
            break
        
        current_primes = list(start_primes)
        current_prod = 1
        for p in current_primes:
            current_prod *= p
        
        if current_prod > LIMIT:
            continue
        
        # Evaluate starting point
        factors = [(p, 1) for p in current_primes]
        score, _ = tau_perp(current_prod, factors)
        if score > best_score:
            best_score = score
            best_n = current_prod
        
        # Iterative local search: try single prime swaps
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for idx in range(len(current_primes)):
                if time.time() - start_time > DEADLINE:
                    return best_n
                old_p = current_primes[idx]
                # Try replacing with primes in a range around old_p
                search_range = [p for p in small_primes if 0.5 * old_p < p < 3 * old_p and p != old_p]
                for new_p in search_range:
                    if new_p in current_primes:
                        continue
                    new_prod = current_prod // old_p * new_p
                    if new_prod > LIMIT:
                        continue
                    new_primes = current_primes[:idx] + [new_p] + current_primes[idx+1:]
                    new_primes.sort()
                    factors = [(p, 1) for p in new_primes]
                    score, _ = tau_perp(new_prod, factors)
                    if score > best_score:
                        best_score = score
                        best_n = new_prod
                        current_primes = new_primes
                        current_prod = new_prod
                        improved = True
                        break

    if time.time() - start_time > DEADLINE:
        return best_n

    # --- Phase 5: Try 16 primes if possible ---
    # Product of first 16 primes exceeds 10^18, but maybe a different set?
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47*53 > 10^18
    # So 16 distinct primes won't work with the smallest.
    # But what about 15 primes where we can add one more small factor?
    # That would make it non-squarefree.
    
    # Try: 15 primes + one of them squared
    for i, p in enumerate(primes_15):
        if time.time() - start_time > DEADLINE:
            break
        n_test = n_base * p
        if n_test > LIMIT:
            continue
        factors = [(pp, 2 if pp == p else 1) for pp in primes_15]
        score, _ = tau_perp(n_test, factors)
        if score > best_score:
            best_score = score
            best_n = n_test

    if time.time() - start_time > DEADLINE:
        return best_n

    # --- Phase 6: Genetic Algorithm on prime exponent vectors ---
    # Model: each individual is a list of (prime, exponent) pairs.
    # Fitness = tau_perp score.
    # Mutation: swap primes, adjust exponents, guided by log-budget constraint.
    # The log-budget is sum(e_i * log(p_i)) <= log(10^18).
    # We promote formation of '1' bits (coprime adjacencies) by favoring
    # configurations with many distinct small primes (high support-disjoint boundaries).
    
    random.seed(42)
    
    def log_budget(factors):
        """Compute log of the number represented by factors."""
        return sum(e * math.log(p) for p, e in factors)
    
    def log_budget_limit():
        return math.log(10**18)
    
    def fitness_from_factors(factors):
        """Compute fitness (score) from factor list, or -inf if over budget."""
        if log_budget(factors) > log_budget_limit() + 1e-9:
            return float('-inf')
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT or n < 2:
            return float('-inf')
        score, _ = tau_perp(n, factors)
        return score
    
    def mutate_factors(factors, available_primes):
        """Mutate a factor vector: swap primes, adjust exponents."""
        new_factors = [(p, e) for p, e in factors]
        mutation_type = random.random()
        
        if mutation_type < 0.4 and len(new_factors) > 2:
            # Swap one prime with another
            idx = random.randint(0, len(new_factors) - 1)
            old_p, old_e = new_factors[idx]
            candidates = [p for p in available_primes if p != old_p]
            if candidates:
                new_p = random.choice(candidates)
                new_factors[idx] = (new_p, old_e)
        elif mutation_type < 0.6:
            # Change exponent of a random prime
            if new_factors:
                idx = random.randint(0, len(new_factors) - 1)
                p, e = new_factors[idx]
                new_e = max(1, min(4, e + random.choice([-1, 0, 1])))
                new_factors[idx] = (p, new_e)
        elif mutation_type < 0.8:
            # Remove a prime (if we have enough)
            if len(new_factors) > 3:
                idx = random.randint(0, len(new_factors) - 1)
                new_factors.pop(idx)
        else:
            # Add a new prime with exponent 1
            candidates = [p for p in available_primes if p not in [pf[0] for pf in new_factors]]
            if candidates:
                new_factors.append((random.choice(candidates), 1))
        
        new_factors.sort(key=lambda x: x[0])
        return new_factors
    
    def crossover_factors(f1, f2):
        """Crossover: take a random prefix from f1 and suffix from f2."""
        if len(f1) == 0 or len(f2) == 0:
            return list(f1) if f1 else list(f2)
        cut = random.randint(1, min(len(f1), len(f2)))
        child = list(f1[:cut]) + list(f2[cut:])
        # Remove duplicates by prime
        seen = set()
        unique = []
        for p, e in child:
            if p not in seen:
                seen.add(p)
                unique.append((p, e))
        unique.sort(key=lambda x: x[0])
        return unique
    
    # Genetic algorithm setup
    ga_available_primes = [p for p in small_primes if p <= 200]
    ga_pop_size = 80
    ga_generations = 300
    
    # Initialize population
    ga_population = []
    
    # Seed with known good configurations
    base_configs = [
        [(p, 1) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]],
        [(p, 1) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 53]],
        [(p, 1) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 53]],
        [(2, 2)] + [(p, 1) for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]],
        [(p, 1) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]] + [(47, 2)],
    ]
    
    for cfg in base_configs:
        if log_budget(cfg) <= log_budget_limit():
            ga_population.append(cfg)
    
    # Fill rest with random valid configurations
    while len(ga_population) < ga_pop_size:
        n_primes = random.randint(12, 16)
        primes_to_use = random.sample(ga_available_primes, min(n_primes, len(ga_available_primes)))
        factors = [(p, 1) for p in primes_to_use]
        # Randomly bump some exponents
        for _ in range(random.randint(0, 3)):
            if factors:
                idx = random.randint(0, len(factors) - 1)
                p, e = factors[idx]
                factors[idx] = (p, min(4, e + 1))
        factors.sort(key=lambda x: x[0])
        if log_budget(factors) <= log_budget_limit():
            ga_population.append(factors)
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # Run genetic algorithm
    for gen in range(ga_generations):
        if time.time() - start_time > DEADLINE:
            break
        
        # Evaluate fitness
        fitnesses = [fitness_from_factors(ind) for ind in ga_population]
        
        # Update best
        max_fit_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[max_fit_idx] > best_score:
            best_score = int(fitnesses[max_fit_idx])
            best_n = 1
            for p, e in ga_population[max_fit_idx]:
                best_n *= p ** e
        
        # Selection: tournament
        def tournament_select():
            k = 3
            indices = [random.randint(0, len(fitnesses) - 1) for _ in range(k)]
            return max(indices, key=lambda i: fitnesses[i])
        
        # Create new population
        new_population = []
        
        # Elitism: keep top 2
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: -fitnesses[i])
        for i in sorted_indices[:2]:
            if fitnesses[i] > float('-inf'):
                new_population.append(list(ga_population[i]))
        
        # Fill rest with offspring
        while len(new_population) < ga_pop_size:
            if time.time() - start_time > DEADLINE:
                break
            parent1_idx = tournament_select()
            parent2_idx = tournament_select()
            
            if random.random() < 0.5:
                child = crossover_factors(ga_population[parent1_idx], ga_population[parent2_idx])
            else:
                child = list(ga_population[parent1_idx])
            
            # Mutate
            child = mutate_factors(child, ga_available_primes)
            
            if child:
                new_population.append(child)
        
        ga_population = new_population
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # --- Phase 7: Final polish with more aggressive exploration ---
    # Try various perturbations of the best found number
    if best_n > 1:
        # Try multiplying/dividing by small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]:
            if time.time() - start_time > DEADLINE:
                break
            n_test = best_n * p
            if n_test <= LIMIT:
                factors = None
                score, _ = tau_perp(n_test, factors)
                if score > best_score:
                    best_score = score
                    best_n = n_test
            if best_n % p == 0:
                n_test2 = best_n // p
                if n_test2 >= 2:
                    factors = None
                    score, _ = tau_perp(n_test2, factors)
                    if score > best_score:
                        best_score = score
                        best_n = n_test2

        # Try swapping one prime factor with another
        factors_list = get_prime_set_from_n(best_n)
        if len(factors_list) >= 10:
            for i in range(len(factors_list)):
                if time.time() - start_time > DEADLINE:
                    break
                old_p = factors_list[i]
                for new_p in small_primes:
                    if new_p == old_p or new_p in factors_list:
                        continue
                    new_prod = best_n // old_p * new_p
                    if new_prod > LIMIT:
                        continue
                    new_factors = factors_list[:i] + [new_p] + factors_list[i+1:]
                    new_factors.sort()
                    factors = [(p, 1) for p in new_factors]
                    score, _ = tau_perp(new_prod, factors)
                    if score > best_score:
                        best_score = score
                        best_n = new_prod
                        factors_list = new_factors

    return best_n
