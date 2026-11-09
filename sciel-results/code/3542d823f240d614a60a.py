import time

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute smallest prime factor (SPF) up to max B
    max_B = 256 * 192 + 100
    spf = list(range(max_B + 1))
    for i in range(2, int(max_B**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_B + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Factorize using SPF table."""
        factors = {}
        while num > 1:
            p = spf[num]
            cnt = 0
            while num % p == 0:
                num //= p
                cnt += 1
            factors[p] = cnt
        return factors
    
    factor_cache = {}
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
            
            B_sq = B * B
            neg_B_mod_A = (-B) % A
            
            if B in factor_cache:
                factors = factor_cache[B]
            else:
                factors = factorize(B)
                factor_cache[B] = factors
            
            # Generate divisors of B^2
            divs = [1]
            for p, e in factors.items():
                new_divs = []
                pe = 2 * e
                pow_p = 1
                for i in range(pe + 1):
                    for d in divs:
                        new_divs.append(d * pow_p)
                    pow_p *= p
                divs = new_divs
            
            found_sol = None
            for u in divs:
                if u % A == neg_B_mod_A:
                    num_y = u + B
                    if num_y % A != 0:
                        continue
                    y = num_y // A
                    
                    v = B_sq // u
                    num_z = v + B
                    if num_z % A != 0:
                        continue
                    z = num_z // A
                    
                    if x == y or x == z or y == z:
                        continue
                    
                    if y > 10**12 or z > 10**12:
                        continue
                    
                    found_sol = [x, y, z]
                    break
            
            if found_sol:
                found = found_sol
                break
        
        answers.append(found)
        
    return answers
