import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    # Primes to consider. Product of first 15 primes is ~6.1e17.
    # Including 16th prime (67) makes product ~4e19 > 1e18.
    # We focus on square-free structures primarily, as they maximize divisor count per bit of budget.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    k = len(primes)
    LIMIT = 10**18

    def calc_score(exps):
        """
        Calculate tau_perp(n) for n = prod p_i^e_i.
        Returns (score, n).
        """
        n = 1
        divisors = [1]
        
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            n *= (p ** e)
            if n > LIMIT:
                return -1, n
            
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divisors = new_divs
            
        if n < 2:
            return 0, n
            
        divisors.sort()
        
        score = 0
        prev = divisors[0]
        for i in range(1, len(divisors)):
            curr = divisors[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
            
        return score, n

    # Strategy:
    # 1. Start with the largest possible square-free number (primorial).
    # 2. The score for square-free numbers is high because divisors are products of distinct primes.
    # 3. We can potentially increase the score by adding exponents to smaller primes?
    #    Actually, increasing exponents increases the number of divisors, which generally increases the chance of coprime adjacencies, 
    #    BUT it also makes the numbers denser, potentially reducing coprime gaps?
    #    Empirically, square-free numbers with many primes tend to be very good.
    #    Let's test: Is n = 2*3*5*...*47 (15 primes) the best?
    #    Score was ~21.9.
    #    
    #    Let's try a different approach:
    #    Instead of random walk, let's do a systematic local search around the primorial.
    #    We will try:
    #    - Removing a large prime and adding a small prime squared? (e.g., remove 47, add 2^2)
    #    - Removing a large prime and adding a medium prime?
    #    
    #    Actually, the "greedy interval-merging" idea suggests we should look at the structure of the divisor lattice.
    #    For square-free n, divisors are all subsets. 
    #    Adjacent divisors d, d' in sorted order.
    #    gcd(d, d') = 1 if and only if they share no common prime factor.
    #    
    #    Let's implement a fast local search that specifically targets "swaps" of prime factors.
    #    State: set of active primes (exponent 1) and possibly some with exponent > 1.
    #    
    #    Let's start with the maximal square-free set.
    best_exps = [0] * k
    n_current = 1
    for i, p in enumerate(primes):
        if n_current * p <= LIMIT:
            best_exps[i] = 1
            n_current *= p
        else:
            break
            
    best_score, best_n = calc_score(best_exps)
    
    # Local Search:
    # 1. Try increasing exponent of a small prime if it fits.
    # 2. Try swapping a large prime for a smaller prime's power.
    
    # Precompute powers for speed
    prime_powers = []
    for i, p in enumerate(primes):
        powers = [1]
        val = 1
        while val * p <= LIMIT:
            val *= p
            powers.append(val)
        prime_powers.append(powers)
        
    # To make search faster, we limit the search to exponents 0, 1, 2 for most, maybe 3 for 2.
    # Max exponent for 2 is 59, but 2^10 = 1024.
    
    # We will perform a limited number of iterations of:
    # - Pick a random prime index i
    # - Pick a random new exponent e' in [0, max_allowed]
    # - If n_new <= LIMIT, evaluate score.
    # - If better, accept.
    
    # To avoid the slow O(D log D) sort every time, we can try to be smarter, 
    # but with 120s, even 1000 evaluations of 100k divisors is doable if optimized.
    # 100k sort is ~0.02s. 1000 * 0.02 = 20s.
    # So we can do ~4000-5000 evaluations.
    
    random.seed(42)
    
    # Define a "smart" move generator
    # Moves:
    # 1. Increase exp of p_i by 1 (if fits)
    # 2. Decrease exp of p_i by 1
    # 3. Swap: decrease exp of p_j (large) by 1, increase exp of p_i (small) by 1
    # 4. Random restart of exponents for a subset
    
    current_exps = best_exps[:]
    current_score = best_score
    current_n = best_n
    
    # We'll use a simple simulated annealing but with structured moves.
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    # Track recent best to allow restarts if stuck
    last_improvement = time.time()
    
    while time.time() < DEADLINE:
        # Generate move
        move_type = random.random()
        
        new_exps = current_exps[:]
        
        if move_type < 0.5:
            # Single change
            i = random.randint(0, k - 1)
            delta = random.choice([-1, 1])
            if random.random() < 0.2:
                delta = random.choice([-2, 2])
            new_exps[i] += delta
            if new_exps[i] < 0:
                new_exps[i] = 0
            # Cap exponent to avoid huge numbers unnecessarily
            if new_exps[i] > 10: # 2^10 is small, but for larger primes, keep it low
                new_exps[i] = 10
        elif move_type < 0.8:
            # Swap: remove one from large prime, add one to small prime
            # Pick a prime to remove from (prefer larger primes)
            candidates_remove = [i for i in range(k) if current_exps[i] > 0]
            if not candidates_remove:
                continue
            # Weight towards larger primes
            weights = [primes[i] for i in candidates_remove]
            i_rem = random.choices(candidates_remove, weights=weights, k=1)[0]
            
            # Pick a prime to add to (prefer smaller primes)
            candidates_add = [i for i in range(k)]
            weights_add = [1.0 / primes[i] for i in candidates_add]
            i_add = random.choices(candidates_add, weights=weights_add, k=1)[0]
            
            new_exps[i_rem] -= 1
            new_exps[i_add] += 1
            if new_exps[i_rem] < 0:
                new_exps[i_rem] = 0
            if new_exps[i_add] > 10:
                new_exps[i_add] = 10
        else:
            # Random perturbation: change 2-3 exponents
            for _ in range(random.randint(2, 3)):
                i = random.randint(0, k - 1)
                new_exps[i] += random.choice([-1, 1])
                if new_exps[i] < 0:
                    new_exps[i] = 0
                if new_exps[i] > 10:
                    new_exps[i] = 10

        # Check validity and score
        # Quick check: does it fit?
        n_new = 1
        valid = True
        for i, e in enumerate(new_exps):
            if e > 0:
                n_new *= (primes[i] ** e)
                if n_new > LIMIT:
                    valid = False
                    break
        
        if not valid:
            continue
            
        score_new, _ = calc_score(new_exps)
        
        if score_new > current_score:
            current_score = score_new
            current_n = n_new
            current_exps = new_exps
            if current_score > best_score:
                best_score = current_score
                best_n = current_n
                best_exps = new_exps[:]
                last_improvement = time.time()
        else:
            # Accept worse with probability
            delta_score = score_new - current_score
            if delta_score == 0:
                if random.random() < 0.5:
                    current_score = score_new
                    current_n = n_new
                    current_exps = new_exps
            else:
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    current_score = score_new
                    current_n = n_new
                    current_exps = new_exps
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        # If stuck for too long, restart with a new random state?
        # Or just keep going.
        
    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
