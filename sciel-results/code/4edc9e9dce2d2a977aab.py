import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Primes up to 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def get_divisors(n):
        divs = [1]
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                new_divs = []
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            if temp == 1:
                break
        if temp > 1:
            for d in divs[:]:
                divs.append(d * temp)
        return sorted(divs)

    def calc_score(n):
        if n < 2:
            return 0
        divs = get_divisors(n)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    candidates = set()
    
    # Base products
    # P14: 2*3*...*43
    p14 = 1
    for p in primes[:14]:
        p14 *= p
        
    # P13: 2*3*...*41
    p13 = p14 // 43
    
    # P12: 2*3*...*37
    p12 = p13 // 41
    
    # Generate candidates based on analysis
    # P14 base (14 distinct primes)
    # Max multiplier ~76. Best: 2^5 3^2 (mult 48), 2^4 3^3 (mult 72)
    c1 = p14 * (2**4) * (3**2) # 2^5 * 3^3 ... wait. Base is 2^1 3^1.
    # If I multiply by 2^4 3^2, I get 2^5 3^3.
    # Let's stick to explicit exponents to avoid confusion.
    
    def make_n(exp_2, exp_3, exp_5, exp_7, num_primes):
        # num_primes is count of primes from 2..
        # e.g., 14 means 2..43
        n = 1
        for i, p in enumerate(primes[:num_primes]):
            if i == 0:
                n *= p ** exp_2
            elif i == 1:
                n *= p ** exp_3
            elif i == 2:
                n *= p ** exp_5
            elif i == 3:
                n *= p ** exp_7
            else:
                n *= p
        return n

    # P14 candidates
    # 2^5 3^2 5^1 ... 43^1
    n1 = make_n(5, 2, 1, 1, 14)
    if n1 <= 10**18: candidates.add(n1)
    
    # 2^4 3^3 5^1 ... 43^1
    n2 = make_n(4, 3, 1, 1, 14)
    if n2 <= 10**18: candidates.add(n2)
    
    # 2^6 3^1 5^1 ... 43^1
    n3 = make_n(6, 1, 1, 1, 14)
    if n3 <= 10**18: candidates.add(n3)
    
    # 2^3 3^4 5^1 ... 43^1
    n4 = make_n(3, 4, 1, 1, 14)
    if n4 <= 10**18: candidates.add(n4)
    
    # 2^2 3^5 5^1 ... 43^1
    n5 = make_n(2, 5, 1, 1, 14)
    if n5 <= 10**18: candidates.add(n5)
    
    # P13 candidates (13 distinct primes, 2..41)
    # Best was 2^11 3^2
    n6 = make_n(11, 2, 1, 1, 13)
    if n6 <= 10**18: candidates.add(n6)
    
    # 2^10 3^3
    n7 = make_n(10, 3, 1, 1, 13)
    if n7 <= 10**18: candidates.add(n7)
    
    # 2^9 3^4
    n8 = make_n(9, 4, 1, 1, 13)
    if n8 <= 10**18: candidates.add(n8)
    
    # 2^8 3^5
    n9 = make_n(8, 5, 1, 1, 13)
    if n9 <= 10**18: candidates.add(n9)
    
    # 2^7 3^6
    n10 = make_n(7, 6, 1, 1, 13)
    if n10 <= 10**18: candidates.add(n10)
    
    # P12 candidates (12 distinct primes, 2..37)
    # Best was 2^7 3^5 5^3
    n11 = make_n(7, 5, 3, 1, 12)
    if n11 <= 10**18: candidates.add(n11)
    
    # 2^8 3^4 5^3
    n12 = make_n(8, 4, 3, 1, 12)
    if n12 <= 10**18: candidates.add(n12)
    
    # 2^9 3^3 5^3
    n13 = make_n(9, 3, 3, 1, 12)
    if n13 <= 10**18: candidates.add(n13)
    
    # 2^6 3^6 5^3
    n14 = make_n(6, 6, 3, 1, 12)
    if n14 <= 10**18: candidates.add(n14)
    
    # 2^5 3^7 5^3
    n15 = make_n(5, 7, 3, 1, 12)
    if n15 <= 10**18: candidates.add(n15)
    
    # 2^10 3^2 5^4
    n16 = make_n(10, 2, 4, 1, 12)
    if n16 <= 10**18: candidates.add(n16)
    
    # 2^9 3^3 5^4
    n17 = make_n(9, 3, 4, 1, 12)
    if n17 <= 10**18: candidates.add(n17)

    # Square free 15 primes for comparison
    n18 = make_n(1, 1, 1, 1, 15)
    if n18 <= 10**18: candidates.add(n18)
    
    best_n = 2
    best_score = -1
    
    for n in candidates:
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n

    return best_n
