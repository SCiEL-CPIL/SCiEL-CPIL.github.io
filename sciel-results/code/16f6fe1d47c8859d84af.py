import time

# Precompute solutions for n=3..256
_SOLUTIONS = {}

def _find_solution(n):
    # 1/x + 1/y + 1/z = 4/n with x < y < z (WLOG for distinctness)
    # x must satisfy: 4x - n > 0 => x > n/4
    # Also 3/x > 4/n => x < 3n/4 (since 1/x > 1/y > 1/z, so 3/x > 4/n)
    # So x is in (n/4, 3n/4)
    
    x_min = n // 4 + 1
    x_max = (3 * n) // 4 + 1  # Slightly wider for safety
    
    for x in range(x_min, x_max + 1):
        a = 4 * x - n  # Must be > 0
        if a <= 0:
            continue
        b = n * x
        
        # 1/y + 1/z = a/b
        # (ay - b)(az - b) = b^2
        # Let u = ay - b, v = az - b, then u*v = b^2
        # y = (u + b) / a, z = (v + b) / a
        # Need: u + b ≡ 0 (mod a), v + b ≡ 0 (mod a)
        # u, v > 0, u <= v (WLOG)
        
        # y range: y > b/a (so ay > b, i.e., u > 0)
        # y < 2b/a (since 2/y > a/b for y < z)
        # Also y > x for distinctness (assuming x is smallest)
        
        y_lo = max(x + 1, b // a + 1)
        y_hi = min(2 * b // a + 1, 10**12)
        
        if y_hi < y_lo:
            continue
        
        # Iterate y in the valid range
        # 1/z = a/b - 1/y = (ay - b) / (by)
        # z = by / (ay - b)
        # Need (ay - b) | by and z > y, z <= 10^12
        
        for y in range(y_lo, y_hi + 1):
            num = a * y - b
            if num <= 0:
                continue
            den = b * y
            if den % num != 0:
                continue
            z = den // num
            if z <= y:
                continue
            if z > 10**12:
                continue
            # Check all distinct: x < y < z ensures all distinct
            return (x, y, z)
    
    # If not found with x < y < z, try other orderings
    # But for efficiency, the above should cover most cases
    return None

def _precompute():
    global _SOLUTIONS
    _SOLUTIONS = {}
    for n in range(3, 257):
        _SOLUTIONS[n] = _find_solution(n)

# Run precomputation at module load
_start_time = time.time()
_precompute()

def construct():
    answers = []
    for n in range(3, 257):
        sol = _SOLUTIONS.get(n)
        if sol:
            answers.append(list(sol))
        else:
            answers.append(None)
    return answers
