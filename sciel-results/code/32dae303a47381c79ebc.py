import math
import random
from collections import defaultdict

def is_prime(n):
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def pollard_rho(n):
    if n % 2 == 0: return 2
    if n % 3 == 0: return 3
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factorize(n):
    factors = []
    def _factor(n):
        if n == 1: return
        if is_prime(n):
            factors.append(n)
        else:
            d = pollard_rho(n)
            _factor(d)
            _factor(n // d)
    _factor(n)
    f_dict = defaultdict(int)
    for p in factors:
        f_dict[p] += 1
    return f_dict

def get_divisors(f_dict):
    divisors = [1]
    for p, e in f_dict.items():
        new_divs = []
        p_power = 1
        for i in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_power)
            p_power *= p
        divisors = new_divs
    return divisors

def sigma_from_factorization(f_dict):
    sigma = 1
    for p, e in f_dict.items():
        term = 1
        p_power = 1
        for i in range(e):
            p_power *= p
            term += p_power
        sigma *= term
    return sigma

def factorize_k_using_M_primes(k, M_primes):
    f_dict = defaultdict(int)
    for p in M_primes:
        while k % p == 0:
            k //= p
            f_dict[p] += 1
    if k > 1:
        f_dict[k] += 1
    return f_dict

def construct():
    random.seed(42)
    candidates = set()
    
    # Generate random smooth numbers
    for _ in range(1000):
        a = random.randint(10, 60)
        b = random.randint(5, 30) if random.random() > 0.5 else 0
        c = random.randint(3, 15) if random.random() > 0.5 else 0
        d = random.randint(2, 10) if random.random() > 0.5 else 0
        e = random.randint(1, 5) if random.random() > 0.5 else 0
        f = random.randint(1, 3) if random.random() > 0.5 else 0
        
        M = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d) * (11 ** e) * (13 ** f)
        if M > 10**18:
            continue
        candidates.add(M)
        
    # Add some specific highly composite-like numbers
    base = 963761198400
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if base * p <= 10**18:
            candidates.add(base * p)
        if base * p * p <= 10**18:
            candidates.add(base * p * p)
            
    best_M = None
    best_ks = []
    
    for M in candidates:
        if M > 10**18:
            continue
            
        M_f_dict = factorize(M)
        M_primes = list(M_f_dict.keys())
        divisors = get_divisors(M_f_dict)
        
        valid_ks = []
        sqrt_M = math.isqrt(M)
        
        for k in divisors:
            if k > sqrt_M:
                continue
            
            k_f_dict = factorize_k_using_M_primes(k, M_primes)
            s = sigma_from_factorization(k_f_dict)
            
            if k * s == M:
                valid_ks.append(k)
                
        if len(valid_ks) > len(best_ks):
            best_ks = valid_ks
            best_M = M
            
    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
