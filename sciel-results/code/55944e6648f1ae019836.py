import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
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

    candidates = set()
    
    # P14 candidates (14 distinct primes, 2..43)
    # Base product of 2..43 is ~1.3e17. Max multiplier ~76.
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
    
    # 2^1 3^6 5^1 ... 43^1
    n6 = make_n(1, 6, 1, 1, 14)
    if n6 <= 10**18: candidates.add(n6)
    
    # 2^5 3^1 5^2 7^1 ... 43^1
    n7 = make_n(5, 1, 2, 1, 14)
    if n7 <= 10**18: candidates.add(n7)
    
    # 2^4 3^2 5^2 7^1 ... 43^1
    n8 = make_n(4, 2, 2, 1, 14)
    if n8 <= 10**18: candidates.add(n8)
    
    # 2^3 3^3 5^2 7^1 ... 43^1
    n9 = make_n(3, 3, 2, 1, 14)
    if n9 <= 10**18: candidates.add(n9)
    
    # 2^2 3^4 5^2 7^1 ... 43^1
    n10 = make_n(2, 4, 2, 1, 14)
    if n10 <= 10**18: candidates.add(n10)
    
    # 2^1 3^5 5^2 7^1 ... 43^1
    n11 = make_n(1, 5, 2, 1, 14)
    if n11 <= 10**18: candidates.add(n11)
    
    # 2^5 3^2 5^1 7^2 ... 43^1
    n12 = make_n(5, 2, 1, 2, 14)
    if n12 <= 10**18: candidates.add(n12)
    
    # 2^4 3^3 5^1 7^2 ... 43^1
    n13 = make_n(4, 3, 1, 2, 14)
    if n13 <= 10**18: candidates.add(n13)
    
    # 2^3 3^4 5^1 7^2 ... 43^1
    n14 = make_n(3, 4, 1, 2, 14)
    if n14 <= 10**18: candidates.add(n14)
    
    # 2^2 3^5 5^1 7^2 ... 43^1
    n15 = make_n(2, 5, 1, 2, 14)
    if n15 <= 10**18: candidates.add(n15)
    
    # 2^1 3^6 5^1 7^2 ... 43^1
    n16 = make_n(1, 6, 1, 2, 14)
    if n16 <= 10**18: candidates.add(n16)
    
    # P13 candidates (13 distinct primes, 2..41)
    # Base product of 2..41 is ~3e16. Max multiplier ~33333.
    # 2^11 3^2
    n17 = make_n(11, 2, 1, 1, 13)
    if n17 <= 10**18: candidates.add(n17)
    
    # 2^10 3^3
    n18 = make_n(10, 3, 1, 1, 13)
    if n18 <= 10**18: candidates.add(n18)
    
    # 2^9 3^4
    n19 = make_n(9, 4, 1, 1, 13)
    if n19 <= 10**18: candidates.add(n19)
    
    # 2^8 3^5
    n20 = make_n(8, 5, 1, 1, 13)
    if n20 <= 10**18: candidates.add(n20)
    
    # 2^7 3^6
    n21 = make_n(7, 6, 1, 1, 13)
    if n21 <= 10**18: candidates.add(n21)
    
    # 2^6 3^7
    n22 = make_n(6, 7, 1, 1, 13)
    if n22 <= 10**18: candidates.add(n22)
    
    # 2^11 3^1 5^2
    n23 = make_n(11, 1, 2, 1, 13)
    if n23 <= 10**18: candidates.add(n23)
    
    # 2^10 3^2 5^2
    n24 = make_n(10, 2, 2, 1, 13)
    if n24 <= 10**18: candidates.add(n24)
    
    # 2^9 3^3 5^2
    n25 = make_n(9, 3, 2, 1, 13)
    if n25 <= 10**18: candidates.add(n25)
    
    # 2^8 3^4 5^2
    n26 = make_n(8, 4, 2, 1, 13)
    if n26 <= 10**18: candidates.add(n26)
    
    # 2^7 3^5 5^2
    n27 = make_n(7, 5, 2, 1, 13)
    if n27 <= 10**18: candidates.add(n27)
    
    # 2^11 3^1 5^1 7^2
    n28 = make_n(11, 1, 1, 2, 13)
    if n28 <= 10**18: candidates.add(n28)
    
    # 2^10 3^2 5^1 7^2
    n29 = make_n(10, 2, 1, 2, 13)
    if n29 <= 10**18: candidates.add(n29)
    
    # 2^9 3^3 5^1 7^2
    n30 = make_n(9, 3, 1, 2, 13)
    if n30 <= 10**18: candidates.add(n30)
    
    # 2^8 3^4 5^1 7^2
    n31 = make_n(8, 4, 1, 2, 13)
    if n31 <= 10**18: candidates.add(n31)
    
    # 2^7 3^5 5^1 7^2
    n32 = make_n(7, 5, 1, 2, 13)
    if n32 <= 10**18: candidates.add(n32)
    
    # P12 candidates (12 distinct primes, 2..37)
    # Base product of 2..37 is ~7e15. Max multiplier ~1.4e5.
    # 2^7 3^5 5^3
    n33 = make_n(7, 5, 3, 1, 12)
    if n33 <= 10**18: candidates.add(n33)
    
    # 2^8 3^4 5^3
    n34 = make_n(8, 4, 3, 1, 12)
    if n34 <= 10**18: candidates.add(n34)
    
    # 2^9 3^3 5^3
    n35 = make_n(9, 3, 3, 1, 12)
    if n35 <= 10**18: candidates.add(n35)
    
    # 2^6 3^6 5^3
    n36 = make_n(6, 6, 3, 1, 12)
    if n36 <= 10**18: candidates.add(n36)
    
    # 2^5 3^7 5^3
    n37 = make_n(5, 7, 3, 1, 12)
    if n37 <= 10**18: candidates.add(n37)
    
    # 2^7 3^5 5^2 7^2
    n38 = make_n(7, 5, 2, 2, 12)
    if n38 <= 10**18: candidates.add(n38)
    
    # 2^8 3^4 5^2 7^2
    n39 = make_n(8, 4, 2, 2, 12)
    if n39 <= 10**18: candidates.add(n39)
    
    # 2^9 3^3 5^2 7^2
    n40 = make_n(9, 3, 2, 2, 12)
    if n40 <= 10**18: candidates.add(n40)
    
    # 2^6 3^6 5^2 7^2
    n41 = make_n(6, 6, 2, 2, 12)
    if n41 <= 10**18: candidates.add(n41)
    
    # 2^5 3^7 5^2 7^2
    n42 = make_n(5, 7, 2, 2, 12)
    if n42 <= 10**18: candidates.add(n42)
    
    # 2^7 3^5 5^1 7^3
    n43 = make_n(7, 5, 1, 3, 12)
    if n43 <= 10**18: candidates.add(n43)
    
    # 2^8 3^4 5^1 7^3
    n44 = make_n(8, 4, 1, 3, 12)
    if n44 <= 10**18: candidates.add(n44)
    
    # 2^9 3^3 5^1 7^3
    n45 = make_n(9, 3, 1, 3, 12)
    if n45 <= 10**18: candidates.add(n45)
    
    # 2^6 3^6 5^1 7^3
    n46 = make_n(6, 6, 1, 3, 12)
    if n46 <= 10**18: candidates.add(n46)
    
    # 2^5 3^7 5^1 7^3
    n47 = make_n(5, 7, 1, 3, 12)
    if n47 <= 10**18: candidates.add(n47)
    
    # 15 primes (2..47) - likely too big with exponents, but square-free might fit
    # Product of 2..47 is ~6.1e18 > 10^18, so square-free 15 primes doesn't fit.
    # So 15 primes not possible.
    
    best_n = 2
    best_score = 0
    
    # Sort candidates by number of divisors (descending) to prioritize high tau(n)
    def tau(n):
        temp = n
        result = 1
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                e = 0
                while temp % p == 0:
                    temp //= p
                    e += 1
                result *= (e + 1)
            if temp == 1:
                break
        if temp > 1:
            result *= 2
        return result
    
    sorted_candidates = sorted(candidates, key=tau, reverse=True)
    
    for n in sorted_candidates:
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
    
    return best_n
