import math
import time

def score(n):
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    divisors.sort()
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Tail primes 11 to 41
    tail_primes = [11, 13, 17, 19, 23, 29, 31, 37, 41]
    tail_val = 1
    for p in tail_primes:
        tail_val *= p
        
    limit = 10**18 // tail_val
    
    # Candidate 1: 15-primorial
    p15 = 1
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        p15 *= p
    best_n = p15
    best_score = score(p15)
    
    # Search over exponents of 2,3,5,7
    # We want to maximize score, which correlates with tau and coprime density.
    # We will test a grid of exponents.
    # a for 2, b for 3, c for 5, d for 7
    
    # Estimate max exponents
    max_a = int(math.log2(limit))
    max_b = int(math.log(limit, 3))
    max_c = int(math.log(limit, 5))
    max_d = int(math.log(limit, 7))
    
    # To keep it fast, we don't iterate all. We test "balanced" and "max tau" candidates.
    # Max tau usually means small primes have high exponents.
    # We'll test a few specific configurations.
    
    candidates = set()
    candidates.add(p15)
    
    # Add configurations with high powers of 2,3,5,7
    # e.g. a=10, b=6, c=4, d=3
    # We'll just do a nested loop with small steps or specific points.
    
    # Let's try to maximize tau(n) for the prefix 2^a 3^b 5^c 7^d
    # subject to 2^a 3^b 5^c 7^d <= limit
    
    # We can use a simple hill climbing or just test the "canonical" highly composite prefixes.
    # Canonical HCNs: 2^a 3^b 5^c ... with a>=b>=c...
    
    # We'll test a range of 'a' from 0 to max_a, and for each 'a', find best b,c,d.
    # This might be too slow if done naively.
    # But limit is ~10^6.
    # 2^20 is 10^6. So a is up to 20.
    # b is up to 12. c up to 8. d up to 7.
    # Total combos: 20*12*8*7 = 13440.
    # For each, check score? No, score calculation is O(sqrt(n)) or O(tau(n) log tau(n)).
    # tau(n) can be large.
    # We should pre-filter by tau(n) or just test the top 10 by tau(n).
    
    prefix_candidates = []
    for a in range(max_a + 1):
        rem_a = limit // (2**a)
        for b in range(int(math.log(rem_a, 3)) + 1):
            rem_b = rem_a // (3**b)
            for c in range(int(math.log(rem_b, 5)) + 1):
                rem_c = rem_b // (5**c)
                for d in range(int(math.log(rem_c, 7)) + 1):
                    val = (2**a) * (3**b) * (5**c) * (7**d)
                    if val <= limit:
                        tau = (a+1)*(b+1)*(c+1)*(d+1) * (2**9) # 9 tail primes
                        prefix_candidates.append((tau, a, b, c, d, val))
                        
    # Sort by tau descending
    prefix_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Test top 10
    tested = set()
    for i in range(min(10, len(prefix_candidates))):
        if time.time() > deadline:
            break
        tau, a, b, c, d, val = prefix_candidates[i]
        n = val * tail_val
        if n in tested:
            continue
        tested.add(n)
        s = score(n)
        if s > best_score:
            best_score = s
            best_n = n
            
    return best_n
