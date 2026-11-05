import math
import random

def construct():
    # First 20 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    def get_divisors(exponents):
        divs = [1]
        for i, e in enumerate(exponents):
            p = primes[i]
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return divs

    def score(exponents):
        # Check if n <= 10^18
        log_n = 0
        for i, e in enumerate(exponents):
            if e > 0:
                log_n += e * math.log(primes[i])
        if log_n > math.log(10**18) + 1e-9:
            return -1
        
        divs = get_divisors(exponents)
        divs.sort()
        
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count

    # Initial guess: Small exponents for small primes
    # A good starting point is often a highly composite number structure
    # e.g., 2^a * 3^b * 5^c ...
    # Let's try a few random restarts
    
    best_score = 0
    best_exponents = [0] * len(primes)
    
    # Helper to normalize exponents to fit limit
    def scale_exponents(e_list):
        # Scale down if too large
        log_n = sum(e * math.log(primes[i]) for i, e in enumerate(e_list))
        limit = math.log(10**18)
        if log_n > limit:
            factor = limit / log_n
            return [int(e * factor) for e in e_list]
        return e_list

    for restart in range(50):
        # Generate a random initial state
        # Bias towards smaller primes having higher exponents
        exponents = []
        for i in range(len(primes)):
            # Random exponent, decreasing probability for larger primes
            max_exp = max(1, 20 - i)
            e = random.randint(0, max_exp)
            exponents.append(e)
        
        exponents = scale_exponents(exponents)
        
        current_score = score(exponents)
        if current_score > best_score:
            best_score = current_score
            best_exponents = exponents[:]
            
        # Hill climbing
        for step in range(200):
            # Pick a random prime index to modify
            i = random.randint(0, len(primes) - 1)
            # Try +1, -1, or swap with another
            move_type = random.choice(['inc', 'dec', 'swap'])
            
            new_exponents = exponents[:]
            
            if move_type == 'inc':
                if exponents[i] < 60: # Hard cap
                    new_exponents[i] += 1
            elif move_type == 'dec':
                if exponents[i] > 0:
                    new_exponents[i] -= 1
            else:
                j = random.randint(0, len(primes) - 1)
                if i != j:
                    new_exponents[i], new_exponents[j] = new_exponents[j], new_exponents[i]
            
            new_score = score(new_exponents)
            
            if new_score > current_score:
                current_score = new_score
                exponents = new_exponents
                if current_score > best_score:
                    best_score = current_score
                    best_exponents = exponents[:]
            # Simple hill climbing: only accept if better. 
            # For better results, we could use simulated annealing, 
            # but hill climbing with random restarts is robust.
            
    # Reconstruct n
    n = 1
    for i, e in enumerate(best_exponents):
        n *= (primes[i] ** e)
        
    return n
