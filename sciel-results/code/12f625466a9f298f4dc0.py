import math
import random
import time

# EVOLVE-BLOCK-START

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

# First 15 primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
_PRIMES = _sieve(50)
P = len(_PRIMES)  # 15

def _score_exponents(exp):
    """
    Compute tau_perp(n) where n = prod(p_i ** exp[i]).
    Fast method: generate all divisors, sort, count coprime adjacent pairs.
    For n with up to ~10^5 divisors this is fast enough.
    For very large divisor counts we use a smarter approach.
    """
    # Generate divisors from exponents
    divs = [1]
    for i in range(P):
        if exp[i] == 0:
            continue
        p = _PRIMES[i]
        new_divs = []
        pw = 1
        for e in range(exp[i] + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    
    if len(divs) <= 1:
        return 0
    
    divs.sort()
    
    score = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            score += 1
    return score

def _score_exponents_fast(exp):
    """
    Faster scorer using the fact that two divisors are coprime iff
    they share no prime factors.
    We represent each divisor by the set of prime indices it uses.
    But for efficiency with many divisors, we just use the standard method
    with a precomputed gcd trick.
    
    Actually, for the sizes we deal with (tau(n) up to ~10^5),
    generating and sorting divisors + gcd is fast enough in Python.
    Let's optimize by using a bitset representation for the prime support.
    """
    # Build divisor list with their prime-support bitmask
    # Each divisor d = prod p_i^e_i has a bitmask where bit i is set if e_i > 0
    divs = [(1, 0)]  # (value, bitmask)
    for i in range(P):
        if exp[i] == 0:
            continue
        p = _PRIMES[i]
        new_divs = []
        pw = 1
        for e in range(exp[i] + 1):
            mask_bit = (1 << i) if e > 0 else 0
            for (val, mask) in divs:
                new_divs.append((val * pw, mask | mask_bit))
            pw *= p
        divs = new_divs
    
    if len(divs) <= 1:
        return 0
    
    # Sort by value
    divs.sort(key=lambda x: x[0])
    
    score = 0
    for i in range(1, len(divs)):
        # Two divisors are coprime iff their masks are disjoint
        if (divs[i-1][1] & divs[i][1]) == 0:
            score += 1
    return score

def _exponents_to_n(exp):
    n = 1
    for i in range(P):
        if exp[i] > 0:
            n *= _PRIMES[i] ** exp[i]
    return n

def _n_valid(n):
    return 2 <= n <= 10**18

def _max_exp_for_prime(i, current_n):
    """Max exponent for prime i given current product of other primes."""
    if i < 0 or i >= P:
        return 0
    p = _PRIMES[i]
    # Remove current contribution of p
    # This is simplified; we'll handle it in mutation
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    random.seed(42)
    
    # Initialize population with diverse candidates
    # 1. Canonical: first k primes with exponent 1, plus 2^a
    # 2. Random exponent vectors
    # 3. Greedy: add primes one by one if they improve score
    
    population = []
    
    # Canonical candidates
    for k in range(0, 15):
        exp = [0] * P
        for i in range(k):
            exp[i] = 1
        # Try a few exponents of 2
        for a2 in range(0, 8):
            exp[0] = a2
            n = _exponents_to_n(exp)
            if _n_valid(n):
                population.append(exp.copy())
    
    # Random candidates
    for _ in range(30):
        exp = [0] * P
        n = 1
        for i in range(P):
            max_e = int(math.log(10**18 / max(n, 1)) / math.log(_PRIMES[i])) if n > 0 else 60
            max_e = max(0, min(max_e, 10))
            if random.random() < 0.7:
                e = random.randint(1, max(1, max_e))
                exp[i] = e
                n *= _PRIMES[i] ** e
        if _n_valid(_exponents_to_n(exp)):
            population.append(exp)
    
    # Greedy construction
    for _ in range(10):
        exp = [0] * P
        n = 1
        # Randomly order primes
        order = list(range(P))
        random.shuffle(order)
        for i in order:
            p = _PRIMES[i]
            # Try adding this prime with exponent 1 or 2
            for e in [1, 2]:
                new_n = n * (p ** e)
                if new_n <= 10**18:
                    exp[i] = e
                    n = new_n
                    break
        if _n_valid(_exponents_to_n(exp)):
            population.append(exp)
    
    # Deduplicate
    seen = set()
    unique_pop = []
    for exp in population:
        key = tuple(exp)
        if key not in seen:
            seen.add(key)
            unique_pop.append(exp)
    population = unique_pop
    
    # Evaluate initial population
    scored_pop = []
    for exp in population:
        n = _exponents_to_n(exp)
        if not _n_valid(n):
            continue
        s = _score_exponents_fast(exp)
        scored_pop.append((s, exp))
    
    scored_pop.sort(key=lambda x: -x[0])
    
    best_score = scored_pop[0][0] if scored_pop else 0
    best_exp = scored_pop[0][1] if scored_pop else [0] * P
    
    # GA parameters
    pop_size = 60
    elite_size = 10
    mutation_rate = 0.3
    crossover_rate = 0.7
    
    # Fill population to pop_size
    while len(scored_pop) < pop_size:
        # Mutate a random existing individual
        parent = random.choice(scored_pop)[1].copy()
        for i in range(P):
            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    parent[i] = max(0, parent[i] - 1)
                else:
                    parent[i] = min(15, parent[i] + 1)
        n = _exponents_to_n(parent)
        if _n_valid(n):
            s = _score_exponents_fast(parent)
            scored_pop.append((s, parent))
        else:
            scored_pop.append((0, parent))
    
    scored_pop.sort(key=lambda x: -x[0])
    
    generation = 0
    no_improve = 0
    
    while time.time() < deadline:
        generation += 1
        gen_start = time.time()
        
        # Elitism: keep top elite_size
        new_pop = [scored_pop[i].copy() for i in range(min(elite_size, len(scored_pop)))]
        
        # Generate offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline - 5:
                break
            
            if random.random() < crossover_rate and len(scored_pop) >= 2:
                # Tournament selection for two parents
                p1 = max(random.sample(scored_pop, min(3, len(scored_pop))), key=lambda x: x[0])[1]
                p2 = max(random.sample(scored_pop, min(3, len(scored_pop))), key=lambda x: x[0])[1]
                child = [0] * P
                for i in range(P):
                    if random.random() < 0.5:
                        child[i] = p1[i]
                    else:
                        child[i] = p2[i]
            else:
                # Clone a random parent
                parent = max(random.sample(scored_pop, min(3, len(scored_pop))), key=lambda x: x[0])[1]
                child = parent.copy()
            
            # Mutation
            for i in range(P):
                if random.random() < mutation_rate:
                    if random.random() < 0.4:
                        child[i] = max(0, child[i] - 1)
                    elif random.random() < 0.4:
                        child[i] = min(15, child[i] + 1)
                    else:
                        child[i] = random.randint(0, 5)
            
            # Check validity
            n = _exponents_to_n(child)
            if _n_valid(n):
                s = _score_exponents_fast(child)
            else:
                s = 0
            
            new_pop.append((s, child))
        
        # Evaluate and select
        new_pop.sort(key=lambda x: -x[0])
        scored_pop = new_pop[:pop_size]
        
        if scored_pop[0][0] > best_score:
            best_score = scored_pop[0][0]
            best_exp = scored_pop[0][1].copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # If stagnating for many generations, inject random immigrants
        if no_improve >= 15:
            for _ in range(5):
                exp = [0] * P
                n = 1
                for i in range(P):
                    max_e = int(math.log(10**18 / max(n, 1)) / math.log(_PRIMES[i])) if n > 0 else 60
                    max_e = max(0, min(max_e, 10))
                    if random.random() < 0.6:
                        e = random.randint(1, max(1, max_e))
                        exp[i] = e
                        n *= _PRIMES[i] ** e
                if _n_valid(_exponents_to_n(exp)):
                    s = _score_exponents_fast(exp)
                    scored_pop.append((s, exp))
                    if s > best_score:
                        best_score = s
                        best_exp = exp.copy()
            scored_pop.sort(key=lambda x: -x[0])
            scored_pop = scored_pop[:pop_size]
            no_improve = 0
    
    # Final local search around best
    if time.time() < deadline - 5:
        for i in range(P):
            for delta in [-2, -1, 1, 2]:
                trial = best_exp.copy()
                trial[i] = max(0, trial[i] + delta)
                n = _exponents_to_n(trial)
                if _n_valid(n):
                    s = _score_exponents_fast(trial)
                    if s > best_score:
                        best_score = s
                        best_exp = trial
    
    # Convert best exponents to n
    best_n = _exponents_to_n(best_exp)
    if not _n_valid(best_n):
        best_n = 2
    
    return best_n

# EVOLVE-BLOCK-END
