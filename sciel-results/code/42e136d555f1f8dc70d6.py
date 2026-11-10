import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Generate smooth numbers 2^a * 3^b * 5^c * 7^d
    # We want M <= 10^18, so k <= sqrt(M) <= 10^9
    # k is smooth, so k <= 10^9
    max_k = 10**9
    
    candidates = set()
    
    # Powers of 2
    p2 = [1]
    while p2[-1] <= max_k:
        p2.append(p2[-1] * 2)
    p2.pop()
    
    # Powers of 3
    p3 = [1]
    while p3[-1] <= max_k:
        p3.append(p3[-1] * 3)
    p3.pop()
    
    # Powers of 5
    p5 = [1]
    while p5[-1] <= max_k:
        p5.append(p5[-1] * 5)
    p5.pop()
    
    # Powers of 7
    p7 = [1]
    while p7[-1] <= max_k:
        p7.append(p7[-1] * 7)
    p7.pop()
        
    # Generate 7-smooth numbers
    for a in p2:
        if a > max_k: break
        for b in p3:
            ab = a * b
            if ab > max_k: break
            for c in p5:
                abc = ab * c
                if abc > max_k: break
                for d in p7:
                    k = abc * d
                    if k > max_k: break
                    candidates.add(k)
                    
    # For each candidate k, compute M = k * sigma(k)
    # And store in a dict: M -> list of k
    m_to_ks = defaultdict(list)
    
    def sigma(n):
        # Compute sigma(n) for smooth n
        # n = 2^a * 3^b * 5^c * 7^d
        # We can factor n quickly
        res = 1
        for p in [2, 3, 5, 7]:
            if n % p == 0:
                exp = 0
                while n % p == 0:
                    n //= p
                    exp += 1
                res *= (p**(exp+1) - 1) // (p - 1)
        return res

    for k in candidates:
        if time.time() > deadline - 10:
            break
        m = k * sigma(k)
        if m <= 10**18:
            m_to_ks[m].append(k)
            
    # Find M with max solutions
    best_m = 1
    best_ks = [1]
    max_count = 1
    
    for m, ks in m_to_ks.items():
        if len(ks) > max_count:
            max_count = len(ks)
            best_m = m
            best_ks = ks
            
    return {'M': best_m, 'k': best_ks}
