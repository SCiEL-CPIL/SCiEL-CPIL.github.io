import time
import math

# Precompute all solutions at module level
_MAX_B = 256 * 500  # B = n*x, safe upper bound

# Sieve for smallest prime factor
_spf = list(range(_MAX_B + 1))
for _i in range(2, int(_MAX_B ** 0.5) + 1):
    if _spf[_i] == _i:
        for _j in range(_i * _i, _MAX_B + 1, _i):
            if _spf[_j] == _j:
                _spf[_j] = _i

def _factorize(num):
    """Factorize num using precomputed SPF. Returns dict of prime->exponent."""
    if num <= 1:
        return {}
    factors = {}
    while num > 1:
        p = _spf[num]
        cnt = 0
        while num % p == 0:
            num //= p
            cnt += 1
        factors[p] = cnt
    return factors

def _get_divisors_of_square(num):
    """Returns sorted list of divisors d of num^2 where d <= num."""
    factors = _factorize(num)
    divs = [1]
    for p, exp in factors.items():
        new_divs = []
        for d in divs:
            pk = 1
            for k in range(2 * exp + 1):
                new_divs.append(d * pk)
                pk *= p
        divs = new_divs
    divs.sort()
    return [d for d in divs if d <= num]

# Cache for divisor computations
_div_cache = {}

def _get_divisors_cached(num):
    if num not in _div_cache:
        _div_cache[num] = _get_divisors_of_square(num)
    return _div_cache[num]

def _compute_all_solutions():
    """Compute solutions for all n in [3, 256]."""
    answers = []
    for n in range(3, 257):
        found = None
        x_min = n // 4 + 1
        
        # Try x values starting from x_min
        max_x_attempts = 100
        for i in range(max_x_attempts):
            x = x_min + i
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            if B > _MAX_B:
                break
            
            # d_start: need y > x => (d+B)/A > x => d > Ax - B = 4x^2 - 2nx
            d_start = 4 * x * x - 2 * n * x + 1
            if d_start < 1:
                d_start = 1
                
            # Get divisors of B^2 that are <= B
            divs = _get_divisors_cached(B)
            
            for d in divs:
                if d < d_start:
                    continue
                if d > B:
                    break
                    
                num_y = d + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                d_prime = (B * B) // d
                num_z = d_prime + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                if y > x and z > y:
                    found = [x, y, z]
                    break
                    
            if found:
                break
                
        answers.append(found)
    
    return answers

# Precompute at import time
_PRECOMPUTED = _compute_all_solutions()

def construct():
    return _PRECOMPUTED
