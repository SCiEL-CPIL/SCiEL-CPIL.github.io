import time
from math import gcd

def construct():
    answers = [None] * 254
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill (evaluator uses ~60s)
    LIMIT = 1000000000000
    
    # Pre-compute for each n the valid x range and do a smart search
    # We want 1/x + 1/y + 1/z = 4/n with x, y, z distinct positive integers <= 10^12
    
    for idx in range(254):
        n = idx + 3
        if time.time() > deadline:
            continue
        
        found = _find_solution(n, deadline)
        if found is not None:
            answers[idx] = found
    
    return answers

def _find_solution(n, deadline):
    LIMIT = 1000000000000
    
    # We need 1/x + 1/y + 1/z = 4/n
    # Assume x <= y <= z (we'll permute at the end, but we need distinct values)
    # 1/x < 4/n => x > n/4, so x >= n//4 + 1
    # Also 3/x >= 4/n => x <= 3n/4, so x <= 3*n//4 (but we need x such that 4/n - 1/x > 0)
    
    # For a given x, we need 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
    # Let A = 4x - n, B = nx. We need 1/y + 1/z = A/B
    # => (y+z)/(yz) = A/B => B(y+z) = A*yz => A*yz - B*y - B*z = 0
    # => (A*y - B)(A*z - B) = B^2
    # So if we let d = A*y - B, then d divides B^2, and z = (B^2/d + B)/A
    
    # For efficiency, iterate x from a good range.
    # x must satisfy: n/4 < x <= 3n/4 (for the sum to work with positive y,z)
    # Actually more precisely: 1/x < 4/n => x > n/4
    # And 1/x + 1/y + 1/z = 4/n with y,z >= x means 3/x >= 4/n => x <= 3n/4
    
    x_min = n // 4 + 1
    x_max = min(3 * n // 4, LIMIT)
    
    # Try a few strategies:
    # Strategy 1: Try x values and for each, find factor pairs of B^2
    # Strategy 2: Try specific parametric forms
    
    # Let's try x from x_min upward, and for each x, check if we can find y, z
    # For speed, limit the number of x values we try per n
    
    # Actually, let's use the factorization approach more carefully.
    # For given x: A = 4x - n, B = n*x
    # We need d | B^2 where d = A*y - B > 0, and y = (d + B)/A must be integer
    # and z = (B^2//d + B)/A must be integer, with y != x, z != x, y != z
    
    # B^2 can be huge, so we can't enumerate all divisors.
    # Instead, let's just iterate y in a reasonable range for each x.
    
    # Alternative: iterate x, and for each x, iterate y in a bounded range, check z.
    
    # For given x, y must satisfy:
    # 1/y < A/B => y > B/A = nx/(4x-n)
    # 2/y >= A/B (since z >= y) => y <= 2B/A = 2nx/(4x-n)
    
    # So y is in range (B/A, 2B/A] approximately.
    
    # Let's try a limited search: for each n, try x from x_min to x_min + some limit,
    # and for each x, try y from floor(B/A)+1 to min(floor(2B/A)+1, floor(B/A)+limit)
    
    # To make this fast, let's cap the inner loop.
    
    max_x_trials = 200  # number of x values to try per n
    max_y_trials = 200  # number of y values to try per x
    
    # Try x from larger to smaller (gives tighter y range)
    for x in range(x_max, x_max - max_x_trials - 1, -1):
        if time.time() > deadline:
            return None
        
        A = 4 * x - n
        B = n * x
        
        if A <= 0:
            continue
        
        y_low = B // A + 1
        y_high = min(2 * B // A + 1, y_low + max_y_trials)
        
        for y in range(y_low, y_high + 1):
            if y == x:
                continue
            
            # z = B*y / (A*y - B)
            denom = A * y - B
            if denom <= 0:
                continue
            
            numer = B * y
            if numer % denom != 0:
                continue
            
            z = numer // denom
            if z <= 0 or z > LIMIT:
                continue
            if z == x or z == y:
                continue
            
            return [x, y, z]
    
    # If that didn't work, try x from x_min upward
    for x in range(x_min, min(x_min + max_x_trials, x_max + 1)):
        if time.time() > deadline:
            return None
        
        A = 4 * x - n
        B = n * x
        
        if A <= 0:
            continue
        
        y_low = B // A + 1
        y_high = min(2 * B // A + 1, y_low + max_y_trials)
        
        for y in range(y_low, y_high + 1):
            if y == x:
                continue
            
            denom = A * y - B
            if denom <= 0:
                continue
            
            numer = B * y
            if numer % denom != 0:
                continue
            
            z = numer // denom
            if z <= 0 or z > LIMIT:
                continue
            if z == x or z == y:
                continue
            
            return [x, y, z]
    
    # Strategy 3: Use a more clever parametric approach
    # For many n, small solutions exist. Try x = n, then 1/y + 1/z = 3/n
    # => (3y-n)(3z-n) = n^2
    if time.time() < deadline:
        x = n
        A = 3 * n
        B = n * n
        # (A*y - B)(A*z - B) = B^2... wait let me redo
        # 1/y + 1/z = 3/n => n(y+z) = 3yz => 3yz - ny - nz = 0
        # (3y - n)(3z - n) = n^2
        # Let d = 3y - n, then d | n^2, y = (d+n)/3, z = (n^2/d + n)/3
        # Try divisors of n^2
        n2 = n * n
        # Get divisors of n^2
        divs = _divisors(n2)
        for d in divs:
            if d >= n:  # y >= z implies d <= n^2/d, but let's just try all
                pass
            y_num = d + n
            if y_num % 3 != 0:
                continue
            y = y_num // 3
            if y <= 0 or y == x:
                continue
            z_num = n2 // d + n
            if z_num % 3 != 0:
                continue
            z = z_num // 3
            if z <= 0 or z == x or z == y:
                continue
            if z > LIMIT:
                continue
            return [x, y, z]
    
    return None

def _divisors(n):
    """Return all divisors of n."""
    if n <= 0:
        return []
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs)
