import math
import random
from functools import lru_cache

def construct():
    """
    Constructs an integer n <= 10^18 maximizing the number of adjacent coprime divisor pairs.
    Uses a randomized local search over prime factorization exponents.
    """
    LIMIT = 10**18
    
    # Generate a list of small primes
    def generate_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    # We might need primes up to a certain value. 
    # The product of the first 15 primes is ~ 6e17. 
    # If we add powers, we might use smaller primes. 
    # If we use a single large prime, it doesn't help much with divisors count.
    # Let's keep primes up to 200 to be safe for "adding a new prime" moves.
    primes = generate_primes(200)
    
    # Map prime to index for fast access? Or just use the list.
    
    def get_score(n):
        """
        Calculates tau_perp(n): number of adjacent coprime divisor pairs.
        """
        if n == 1:
            return 0
        # Generate divisors
        # We need the factorization. To avoid re-factorizing every time in the search,
        # we should pass the factorization. But for the initial setup and final check, 
        # we can factorize. In the search loop, we maintain the factorization state.
        # However, this function is used for validation.
        
        # For the search, we will have a helper that takes a dict of exponents.
        # But for now, let's make a version that takes n and factorizes it, 
        # or better, let's define the score calculation on the factorization state directly in the search.
        
        # Let's implement a divisor generator from a dict of exponents.
        pass

    def generate_divisors_from_factors(factors):
        """
        factors: dict {prime: exponent}
        Returns sorted list of divisors.
        """
        divisors = [1]
        for p, exp in factors.items():
            current_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divisors:
                    current_divs.append(d * power)
            divisors.extend(current_divs)
        divisors.sort()
        return divisors

    def calculate_score_from_factors(factors):
        """
        Calculates tau_perp for a given factorization state.
        Returns -1 if n > LIMIT.
        """
        n_val = 1
        for p, exp in factors.items():
            if exp > 0:
                n_val *= (p ** exp)
                if n_val > LIMIT:
                    return -1
        
        if n_val < 2:
            return 0
            
        divisors = generate_divisors_from_factors(factors)
        
        score = 0
        # Check adjacent pairs
        # We can optimize: if len(divisors) is large, this loop is O(D log D) for sort and O(D) for check.
        # D is at most ~100k-150k. This is fast.
        
        prev = divisors[0]
        for i in range(1, len(divisors)):
            curr = divisors[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
            
        return score

    # Search State: dict {prime: exponent}
    # Initial Candidates:
    # 1. Primorial (small primes, exp 1)
    # 2. Highly composite-like (higher exponents on small primes)
    # 3. Random valid numbers
    
    best_n = 2
    best_score = 0
    
    # Candidate 1: Primorial
    f1 = {}
    prod = 1
    for p in primes:
        if prod * p <= LIMIT:
            f1[p] = 1
            prod *= p
        else:
            break
    s1 = calculate_score_from_factors(f1)
    if s1 > best_score:
        best_score = s1
        best_n = prod
        
    # Candidate 2: Max exponents on smallest primes
    # Greedy allocation of "budget" to smallest primes
    # Start with 2^a * 3^b * ...
    # We can try a few specific patterns
    patterns = []
    
    # Pattern A: 2^a * 3^b * 5^c ...
    # Try to maximize divisor count first, then refine.
    
    def local_search(initial_factors, iterations=5000):
        """
        Performs hill climbing on the factorization state.
        """
        current_factors = dict(initial_factors)
        current_score = calculate_score_from_factors(current_factors)
        
        if current_score < 0:
            return -1, current_factors
            
        best_local_score = current_score
        best_local_factors = dict(current_factors)
        
        # Helper to get current n value
        def get_n(factors):
            n_val = 1
            for p, exp in factors.items():
                if exp > 0:
                    n_val *= (p ** exp)
            return n_val

        stagnant = 0
        for _ in range(iterations):
            # Generate a move
            # Types of moves:
            # 1. Increase exponent of an existing prime
            # 2. Decrease exponent of an existing prime (if > 1)
            # 3. Add a new prime (exp 1)
            # 4. Remove a prime (set exp 0)
            
            move_type = random.choice([1, 2, 3, 4])
            new_factors = dict(current_factors)
            valid_move = False
            
            if move_type == 1:
                # Increase exponent
                # Choose a prime from current factors
                active_primes = [p for p, e in new_factors.items() if e > 0]
                if active_primes:
                    p = random.choice(active_primes)
                    new_exp = new_factors[p] + 1
                    # Check if n * p <= LIMIT
                    current_n = get_n(new_factors)
                    if current_n * p <= LIMIT:
                        new_factors[p] = new_exp
                        valid_move = True
                    else:
                        # Try a smaller prime? No, we picked random. Just invalid.
                        pass
                        
            elif move_type == 2:
                # Decrease exponent
                active_primes = [p for p, e in new_factors.items() if e > 1]
                if active_primes:
                    p = random.choice(active_primes)
                    new_factors[p] -= 1
                    valid_move = True
                    if new_factors[p] == 0:
                        # We can leave it at 0 or delete. Leaving at 0 is fine for logic, 
                        # but for cleanliness, we might delete later. 
                        # For now, 0 exponent means p^0 = 1, so it doesn't affect n.
                        pass
            elif move_type == 3:
                # Add new prime
                # Choose a prime not in factors (or with 0 exp)
                # Prefer smaller primes to stay within limit
                candidate_primes = [p for p in primes if new_factors.get(p, 0) == 0]
                if candidate_primes:
                    # Bias towards smaller primes
                    # Weighted random choice
                    # Smaller primes are more likely to fit
                    # Simple: pick random from first 20 unused
                    available = [p for p in primes if new_factors.get(p, 0) == 0]
                    if available:
                        # Limit search to smaller primes to avoid huge jumps that fail
                        # Sort by value, pick from first 10 available
                        available.sort()
                        if len(available) > 10:
                            available = available[:10]
                        p = random.choice(available)
                        current_n = get_n(new_factors)
                        if current_n * p <= LIMIT:
                            new_factors[p] = 1
                            valid_move = True
                            
            elif move_type == 4:
                # Remove prime
                active_primes = [p for p, e in new_factors.items() if e > 0]
                if active_primes:
                    p = random.choice(active_primes)
                    new_factors[p] = 0
                    valid_move = True
            
            if not valid_move:
                continue
                
            new_score = calculate_score_from_factors(new_factors)
            
            if new_score > current_score:
                current_factors = new_factors
                current_score = new_score
                stagnant = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_factors = dict(current_factors)
            else:
                stagnant += 1
                # Accept worse moves sometimes? (Simulated Annealing)
                # For now, strict hill climbing is faster. 
                # If stagnant > 100, break?
                if stagnant > 200:
                    break
        
        return best_local_score, best_local_factors

    # Run local search from multiple starting points
    
    # Start 1: Primorial
    f_start_1 = dict(f1)
    s, f_best_1 = local_search(f_start_1, iterations=10000)
    if s > best_score:
        best_score = s
        best_n = get_n(f_best_1) if 'get_n' in locals() else 1 # Recalculate n
        
    # Start 2: "Square-full" like 2^a 3^b ...
    # Let's construct a few more diverse starts
    # Start 2: Max power of 2, then 3, etc.
    f2 = {}
    rem = LIMIT
    for p in primes[:5]: # 2,3,5,7,11
        while rem % (p) == 0: # This logic is wrong, we are constructing, not dividing.
            break
    # Better construction:
    # Try to maximize 2^a * 3^b * 5^c ...
    # Greedy: assign highest exponent to 2, then 3, etc.
    f2 = {}
    current_prod = 1
    for p in primes:
        if p > 100: break
        exp = 0
        while current_prod * (p ** (exp + 1)) <= LIMIT:
            exp += 1
        if exp > 0:
            f2[p] = exp
            current_prod *= (p ** exp)
        # We don't strictly need to fill all, but this gives a "dense" start
    
    s, f_best_2 = local_search(f2, iterations=10000)
    if s > best_score:
        best_score = s
        # Recalculate n from f_best_2
        n_val = 1
        for p, exp in f_best_2.items():
            if exp > 0:
                n_val *= (p ** exp)
        best_n = n_val
        
    # Start 3: Random valid number with many small prime factors
    # Generate random exponents for first 10 primes such that product <= LIMIT
    f3 = {}
    for p in primes[:10]:
        max_exp = 0
        # Estimate max exp based on current prod
        curr = 1
        for pp, ee in f3.items():
            curr *= (pp ** ee)
        while curr * (p ** (max_exp + 1)) <= LIMIT:
            max_exp += 1
        # Random exponent between 1 and max_exp (or 0)
        if max_exp > 0:
            # Bias towards 1 or 2
            f3[p] = random.randint(1, min(5, max_exp))
    
    s, f_best_3 = local_search(f3, iterations=10000)
    if s > best_score:
        best_score = s
        n_val = 1
        for p, exp in f_best_3.items():
            if exp > 0:
                n_val *= (p ** exp)
        best_n = n_val
        
    # Start 4: Just 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53
    # This is similar to primorial but let's ensure we try a few more
    
    # Return the best found
    if best_n < 2:
        return 2
    return best_n

# Helper function needed inside local_search or global?
# The get_n was defined inside local_search in the draft above, 
# but I used it in the main scope. Let's fix the scope.

# Let's rewrite the code cleanly with proper scoping.

def construct():
    LIMIT = 10**18
    
    def generate_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = generate_primes(200)
    
    def get_n_from_factors(factors):
        n_val = 1
        for p, exp in factors.items():
            if exp > 0:
                n_val *= (p ** exp)
        return n_val

    def generate_divisors_from_factors(factors):
        divisors = [1]
        for p, exp in factors.items():
            if exp == 0: continue
            current_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divisors:
                    current_divs.append(d * power)
            divisors.extend(current_divs)
        divisors.sort()
        return divisors

    def calculate_score_from_factors(factors):
        n_val = get_n_from_factors(factors)
        if n_val > LIMIT:
            return -1
        if n_val < 2:
            return 0
            
        divisors = generate_divisors_from_factors(factors)
        
        score = 0
        if len(divisors) < 2:
            return 0
            
        prev = divisors[0]
        for i in range(1, len(divisors)):
            curr = divisors[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
            
        return score

    def local_search(initial_factors, iterations=5000):
        current_factors = dict(initial_factors)
        current_score = calculate_score_from_factors(current_factors)
        
        if current_score < 0:
            return -1, current_factors
            
        best_local_score = current_score
        best_local_factors = dict(current_factors)
        
        stagnant = 0
        for _ in range(iterations):
            move_type = random.choice([1, 2, 3, 4])
            new_factors = dict(current_factors)
            valid_move = False
            
            if move_type == 1:
                active_primes = [p for p, e in new_factors.items() if e > 0]
                if active_primes:
                    p = random.choice(active_primes)
                    new_exp = new_factors[p] + 1
                    current_n = get_n_from_factors(new_factors)
                    if current_n * p <= LIMIT:
                        new_factors[p] = new_exp
                        valid_move = True
                        
            elif move_type == 2:
                active_primes = [p for p, e in new_factors.items() if e > 1]
                if active_primes:
                    p = random.choice(active_primes)
                    new_factors[p] -= 1
                    valid_move = True

            elif move_type == 3:
                available = [p for p in primes if new_factors.get(p, 0) == 0]
                if available:
                    available.sort()
                    if len(available) > 10:
                        available = available[:10]
                    p = random.choice(available)
                    current_n = get_n_from_factors(new_factors)
                    if current_n * p <= LIMIT:
                        new_factors[p] = 1
                        valid_move = True

            elif move_type == 4:
                active_primes = [p for p, e in new_factors.items() if e > 0]
                if active_primes:
                    p = random.choice(active_primes)
                    new_factors[p] = 0
                    valid_move = True
            
            if not valid_move:
                continue
                
            new_score = calculate_score_from_factors(new_factors)
            
            if new_score > current_score:
                current_factors = new_factors
                current_score = new_score
                stagnant = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_factors = dict(current_factors)
            else:
                stagnant += 1
                if stagnant > 200:
                    break
        
        return best_local_score, best_local_factors

    best_n = 2
    best_score = 0
    
    # Start 1: Primorial
    f1 = {}
    prod = 1
    for p in primes:
        if prod * p <= LIMIT:
            f1[p] = 1
            prod *= p
        else:
            break
    s1 = calculate_score_from_factors(f1)
    if s1 > best_score:
        best_score = s1
        best_n = prod
        
    # Start 2: Dense small primes
    f2 = {}
    current_prod = 1
    for p in primes[:10]:
        exp = 0
        while current_prod * (p ** (exp + 1)) <= LIMIT:
            exp += 1
        if exp > 0:
            f2[p] = exp
            current_prod *= (p ** exp)
            
    s2, f_best_2 = local_search(f2, iterations=10000)
    if s2 > best_score:
        best_score = s2
        best_n = get_n_from_factors(f_best_2)
        
    # Start 3: Random
    f3 = {}
    for p in primes[:10]:
        max_exp = 0
        curr = get_n_from_factors(f3)
        while curr * (p ** (max_exp + 1)) <= LIMIT:
            max_exp += 1
        if max_exp > 0:
            f3[p] = random.randint(1, min(5, max_exp))
            
    s3, f_best_3 = local_search(f3, iterations=10000)
    if s3 > best_score:
        best_score = s3
        best_n = get_n_from_factors(f_best_3)
        
    # Start 4: Another random variation
    f4 = {}
    for p in primes[:15]:
        if random.random() < 0.5:
             # Try to add it
             curr = get_n_from_factors(f4)
             if curr * p <= LIMIT:
                 f4[p] = 1
                 
    s4, f_best_4 = local_search(f4, iterations=10000)
    if s4 > best_score:
        best_score = s4
        best_n = get_n_from_factors(f_best_4)
        
    if best_n < 2:
        return 2
    return best_n
