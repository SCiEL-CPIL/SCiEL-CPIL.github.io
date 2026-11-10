import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    # 1. Generate small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # We need primes up to a certain limit. 
    # The product of the first 15 primes is ~6e17.
    # We can use primes up to ~200 to allow for swaps and exponent variations.
    primes = sieve(200)
    K = len(primes)

    # 2. Fast tau_perp calculation given prime factors and exponents
    def tau_perp_from_factors(prime_exp_list):
        # prime_exp_list: list of (p, e)
        if not prime_exp_list:
            return 0
        
        # Generate all divisors
        divs = [1]
        for p, e in prime_exp_list:
            if e == 0:
                continue
            new_divs = []
            power = 1
            for _ in range(e):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs.extend(new_divs)
            
        divs.sort()
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # 3. Initial State
    # Start with the product of the first 15 primes (square-free)
    # But let's try to initialize with some exponents on small primes as suggested by knowledge
    # Product of first 14 primes is ~1.3e17. 2^4 * prod14 = 16 * 1.3e17 = 2.08e18 > 10^18.
    # So 2^3 * prod14 = 8 * 1.3e17 = 1.04e18. Close.
    # Let's start with a mix.
    
    # Initialize exponents to 0
    exps = [0] * K
    current_n = 1
    
    # Fill with first 14 primes
    for i in range(14):
        exps[i] = 1
        current_n *= primes[i]
        
    # Add some exponents to 2 and 3 if they fit
    # current_n is prod(2..43) = 1311395580340863
    # 10^18 / current_n ~ 762
    # We can multiply by 2^9 (512) or 2^10 (1024 > 762). So 2^9 fits.
    # Let's just start with square-free first 15 primes to be safe, then let SA find the rest.
    # Actually, the knowledge says 2^4*prod14 improved. Let's initialize near that.
    
    # Reset
    exps = [0] * K
    current_n = 1
    for i in range(14):
        exps[i] = 1
        current_n *= primes[i]
        
    # Try to add exponents to 2 (index 0)
    while current_n * 2 <= 10**18 and exps[0] < 15:
        exps[0] += 1
        current_n *= 2
        
    best_score = tau_perp_from_factors([(primes[i], exps[i]) for i in range(K) if exps[i] > 0])
    best_exps = exps[:]
    best_n = current_n

    # 4. Simulated Annealing
    # Temperature schedule
    T = 1.0
    cooling_rate = 0.999
    
    # We need to track current state
    cur_exps = best_exps[:]
    cur_n = best_n
    cur_score = best_score
    
    # Precompute log values for speed? No, integer arithmetic is fine.
    
    def get_score(exps):
        # Only include non-zero exponents for speed
        factors = [(primes[i], exps[i]) for i in range(K) if exps[i] > 0]
        return tau_perp_from_factors(factors)

    step_count = 0
    while time.time() < deadline - 5.0:
        T *= cooling_rate
        if T < 0.001:
            T = 0.001
            
        # Choose a random move
        # Move types:
        # 1. Increase exponent of a random prime
        # 2. Decrease exponent of a random prime
        # 3. Swap a prime (remove one, add another) - more complex
        
        move_type = random.random()
        
        new_exps = cur_exps[:]
        new_n = cur_n
        valid_move = False
        delta_score = 0
        
        if move_type < 0.4:
            # Increase exponent
            # Pick a prime that is currently in the set or a small new one
            # To keep n <= 10^18, we must check
            # Prefer increasing exponents of small primes
            candidates = [i for i in range(min(20, K)) if cur_n * primes[i] <= 10**18]
            if candidates:
                i = random.choice(candidates)
                if new_exps[i] < 15: # Cap exponent
                    new_exps[i] += 1
                    new_n *= primes[i]
                    valid_move = True
        elif move_type < 0.8:
            # Decrease exponent
            candidates = [i for i in range(K) if cur_exps[i] > 0]
            if candidates:
                i = random.choice(candidates)
                new_exps[i] -= 1
                new_n //= primes[i]
                valid_move = True
        else:
            # Swap: Remove prime i, add prime j
            # Remove
            rem_candidates = [i for i in range(K) if cur_exps[i] > 0]
            if rem_candidates:
                i = random.choice(rem_candidates)
                # Only allow swap if exponent is 1? Or handle generally?
                # Generally: n_new = n_old / p_i^e_i * p_j^e_j
                # To keep it simple, only swap if e_i == 1 and e_j == 1 (or 0)
                if cur_exps[i] == 1:
                    # Add
                    # Find j such that (n_old / p_i) * p_j <= 10^18
                    max_p = (10**18 * primes[i]) // cur_n
                    add_candidates = [j for j in range(K) if primes[j] <= max_p and cur_exps[j] == 0]
                    if add_candidates:
                        j = random.choice(add_candidates)
                        new_exps[i] = 0
                        new_exps[j] = 1
                        new_n = (cur_n // primes[i]) * primes[j]
                        valid_move = True

        if not valid_move:
            continue
            
        step_count += 1
        
        # Calculate new score
        new_score = get_score(new_exps)
        delta = new_score - cur_score
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept
            cur_exps = new_exps
            cur_n = new_n
            cur_score = new_score
            
            if cur_score > best_score:
                best_score = cur_score
                best_exps = cur_exps[:]
                best_n = cur_n

    return best_n

