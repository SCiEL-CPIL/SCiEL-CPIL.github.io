import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    num_primes = len(primes)
    
    def get_score(exponents):
        """
        Calculate the score for the number defined by exponents.
        """
        divs = [1]
        for i in range(num_primes):
            e = exponents[i]
            if e == 0:
                continue
            p = primes[i]
            new_divs = []
            for d in divs:
                val = 1
                for _ in range(e + 1):
                    new_divs.append(d * val)
                    val *= p
            divs = new_divs
        
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    def calc_n(exponents):
        n = 1
        for i in range(num_primes):
            e = exponents[i]
            if e > 0:
                n *= primes[i] ** e
        return n

    def is_valid(exponents, current_n=None):
        if current_n is not None:
            # We are modifying current_n, so we can just check the ratio
            # But we don't track the exact change in this simple version
            pass
        n = calc_n(exponents)
        return n <= 10**18

    # Initial candidates
    candidates = []
    
    # 1. Primorial P15 (2 to 47)
    e1 = [1]*15 + [0]*3
    candidates.append(e1)
    
    # 2. 2^6 * P14
    e2 = [6] + [1]*13 + [0]*3
    candidates.append(e2)
    
    # 3. 2^5 * 3^2 * P13
    e3 = [5, 2] + [1]*12 + [0]*3
    candidates.append(e3)
    
    # 4. Random valid number
    for _ in range(10):
        exp = [0]*num_primes
        n = 1
        for i in range(num_primes):
            if n > 10**18:
                break
            # Try random exponent
            max_e = int(math.log(10**18 / n, primes[i])) if n < 10**18 else 0
            if max_e > 0:
                e = random.randint(0, min(10, max_e))
                exp[i] = e
                n *= primes[i] ** e
        candidates.append(exp)

    best_score = -1
    best_exponents = None
    best_n = 1
    
    for exp in candidates:
        if time.time() > deadline:
            break
        if is_valid(exp):
            n = calc_n(exp)
            score = get_score(exp)
            if score > best_score:
                best_score = score
                best_exponents = exp[:]
                best_n = n

    # Local search
    current_exp = best_exponents[:]
    current_n = best_n
    
    while time.time() < deadline - 5:
        # Generate a random neighbor
        new_exp = current_exp[:]
        action = random.randint(0, 2)
        
        if action == 0:
            # Increase exponent of a small prime
            i = random.randint(0, min(5, num_primes-1))
            if new_exp[i] < 30:
                new_exp[i] += 1
        elif action == 1:
            # Decrease exponent of a large prime
            i = random.randint(5, num_primes-1)
            if new_exp[i] > 1:
                new_exp[i] -= 1
        else:
            # Swap: increase small, decrease large
            i = random.randint(0, min(5, num_primes-1))
            j = random.randint(5, num_primes-1)
            if i != j and new_exp[j] > 1 and new_exp[i] < 30:
                new_exp[i] += 1
                new_exp[j] -= 1
        
        if not is_valid(new_exp):
            continue
            
        new_n = calc_n(new_exp)
        score = get_score(new_exp)
        
        if score > best_score:
            best_score = score
            best_exponents = new_exp[:]
            best_n = new_n
            current_exp = new_exp[:]
            current_n = new_n

    return best_n
