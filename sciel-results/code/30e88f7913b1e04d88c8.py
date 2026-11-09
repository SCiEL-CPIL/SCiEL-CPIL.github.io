import math
import time
import random

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

# We need primes up to a reasonable limit. 
# Max number of distinct primes for n <= 10^18:
# Product of first k primes <= 10^18
# 2*3*5*...*47 (15 primes) is approx 6e17. 
# 2*3*...*53 (16 primes) is approx 3e19 > 10^18.
# So max 15 distinct primes.
# We'll use the first 20 primes to be safe for potential swaps, but restrict exponents.
_PRIMES = _sieve(100)
# Only keep the first 15 primes as candidates for the main structure, 
# but we might want to consider if a larger prime is better? 
# Usually smaller primes allow higher exponents or more divisors.
# Let's stick to the first 15 primes for the SA state space.
CANDIDATE_PRIMES = _PRIMES[:15]

def _get_divisors(exponents, primes):
    """
    Generate all divisors of n = prod(p_i ^ e_i)
    Returns a sorted list of divisors.
    """
    divs = [1]
    for i, p in enumerate(primes):
        e = exponents[i]
        if e == 0:
            continue
        powers = [1]
        for _ in range(e):
            powers.append(powers[-1] * p)
        new_divs = []
        for d in divs:
            for pw in powers:
                new_divs.append(d * pw)
        divs = new_divs
    divs.sort()
    return divs

def _score_from_divs(divs):
    """
    Count adjacent coprime pairs in a sorted list of divisors.
    """
    score = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            score += 1
    return score

def _score(exponents, primes):
    """
    Calculate score for given exponents.
    Returns 0 if n > 10^18 or n < 2.
    """
    n = 1
    for i, p in enumerate(primes):
        if exponents[i] > 0:
            # Check overflow
            if n > (10**18) // (p ** exponents[i]):
                return 0
            n *= p ** exponents[i]
    
    if n < 2 or n > 10**18:
        return 0
        
    divs = _get_divisors(exponents, primes)
    return _score_from_divs(divs)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial state: Heuristic start
    # Try a few simple initializations
    init_exps = [0] * len(CANDIDATE_PRIMES)
    
    # A good initial guess is often maximizing the number of divisors (tau(n))
    # which tends to correlate with more adjacent pairs.
    # Greedy fill: assign exponents to smallest primes to maximize tau(n) under constraint.
    # Or just start with the "canonical" small prime set with exponents 1.
    # Let's start with exponents = [1, 1, ..., 1] for as many primes as fit.
    
    n_curr = 1
    for i, p in enumerate(CANDIDATE_PRIMES):
        if n_curr * p <= 10**18:
            init_exps[i] = 1
            n_curr *= p
        else:
            break
            
    current_exps = init_exps[:]
    current_score = _score(current_exps, CANDIDATE_PRIMES)
    
    best_exps = current_exps[:]
    best_score = current_score
    best_n = 1
    for i, p in enumerate(CANDIDATE_PRIMES):
        best_n *= p ** best_exps[i]
        
    # Simulated Annealing Parameters
    T = 50.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    max_steps = 100000
    
    step = 0
    while time.time() < deadline and step < max_steps:
        if T < T_min:
            break
            
        # Propose a move:
        # 1. Pick a random prime index
        # 2. Change its exponent by -1, 0, or +1 (weighted towards +1 if small, -1 if large?)
        #    Simple: delta in {-1, 1}
        
        i = random.randint(0, len(CANDIDATE_PRIMES) - 1)
        delta = random.choice([-1, 1])
        
        new_exps = current_exps[:]
        new_exps[i] += delta
        
        # Basic validity check: exponents must be non-negative
        if new_exps[i] < 0:
            continue
            
        # Calculate new score
        new_score = _score(new_exps, CANDIDATE_PRIMES)
        
        if new_score == 0:
            # Invalid n (overflow or < 2)
            # Reject with high probability, or just reject
            continue
            
        delta_score = new_score - current_score
        
        # Acceptance criterion
        if delta_score > 0:
            accept = True
        else:
            # Boltzmann probability
            # If delta_score is 0, accept with some prob to allow drift?
            # Usually accept 0-delta moves to explore plateaus.
            if delta_score == 0:
                accept = random.random() < 0.5
            else:
                # Avoid division by zero or underflow issues
                # exp(delta_score / T)
                # delta_score is negative integer.
                try:
                    prob = math.exp(delta_score / T)
                except OverflowError:
                    prob = 0.0
                except:
                    prob = 0.0
                accept = random.random() < prob
        
        if accept:
            current_exps = new_exps
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_exps = current_exps[:]
                # Calculate best_n
                best_n = 1
                for j, p in enumerate(CANDIDATE_PRIMES):
                    best_n *= p ** best_exps[j]
                
        # Cool down
        T *= cooling_rate
        step += 1
        
    # Final check: ensure best_n is valid
    if best_n < 2 or best_n > 10**18:
        # Fallback to a simple valid number
        best_n = 2
        best_score = 1 # gcd(1,2)=1
        
    return best_n

# EVOLVE-BLOCK-END
