import random
import math
import time

def construct():
    limit = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    best_n = 2
    best_score = -1
    
    # Helper to get divisors
    def get_divisors(n):
        divs = [1]
        for p in primes:
            if n % (p**100) == 0: # Check if p is a factor
                pass
            # Better: factorize n first
        # Simple factorization
        temp_n = n
        divs = [1]
        i = 2
        while i * i <= temp_n:
            if temp_n % i == 0:
                count = 0
                while temp_n % i == 0:
                    temp_n //= i
                    count += 1
                new_divs = []
                for d in divs:
                    for k in range(count + 1):
                        new_divs.append(d * (i**k))
                divs = new_divs
            i += 1 if i == 2 else 2 # Skip even numbers after 2
        if temp_n > 1:
            new_divs = []
            for d in divs:
                new_divs.append(d)
                new_divs.append(d * temp_n)
            divs = new_divs
        return sorted(divs)

    def calc_score(n):
        divs = get_divisors(n)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Time limit check
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    # Initial guess: Primorial
    n = 1
    for p in primes:
        if n * p <= limit:
            n *= p
        else:
            break
    best_n = n
    best_score = calc_score(n)
    
    # Random search
    for _ in range(500):
        if time.time() - start_time > max_time:
            break
        
        # Generate random exponents
        exponents = [0] * len(primes)
        current_n = 1
        for i, p in enumerate(primes):
            # Max exponent for p
            max_e = int(math.log(limit // current_n, p)) if current_n < limit else 0
            if max_e < 0:
                max_e = 0
            exponents[i] = random.randint(0, max_e)
            current_n *= p ** exponents[i]
            
        if current_n > limit:
            continue
            
        # Local search
        for step in range(20):
            if time.time() - start_time > max_time:
                break
            score = calc_score(current_n)
            if score > best_score:
                best_score = score
                best_n = current_n
            
            # Pick a random prime to modify
            idx = random.randint(0, len(primes)-1)
            p = primes[idx]
            old_e = exponents[idx]
            
            # Try +1
            if current_n * p <= limit:
                exponents[idx] = old_e + 1
                current_n *= p
                new_score = calc_score(current_n)
                if new_score >= score:
                    score = new_score
                else:
                    exponents[idx] = old_e
                    current_n //= p
            
            # Try -1
            if old_e > 0:
                exponents[idx] = old_e - 1
                current_n //= p
                new_score = calc_score(current_n)
                if new_score >= score:
                    score = new_score
                else:
                    exponents[idx] = old_e
                    current_n *= p
        
    return best_n
