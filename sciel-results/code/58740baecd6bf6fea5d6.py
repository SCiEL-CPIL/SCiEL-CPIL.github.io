import time

# Precompute solutions for n=3..256
_SOLUTIONS = {}
_DEADLINE = 85.0  # seconds, with safety margin

def _find_solution(n):
    """Find distinct positive integers x, y, z <= 10^12 such that 1/x+1/y+1/z = 4/n."""
    # Assume x <= y <= z (we'll enforce distinctness separately)
    # Then 3/x >= 4/n => x <= 3n/4
    # And 1/x < 4/n => x > n/4
    # So x in (n/4, 3n/4]
    
    start_x = n // 4 + 1
    # Upper bound: x must satisfy 1/x + 2/x >= 4/n => 3/x >= 4/n => x <= 3n/4
    # But we also need x, y, z distinct, so let's be a bit more generous
    end_x = (3 * n) // 4 + 1
    
    for x in range(start_x, end_x + 1):
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        
        # 1/y + 1/z = A/B
        # Since y >= x, we have 1/y <= 1/x, so 1/z >= A/B - 1/x
        # Also 1/y + 1/z = A/B implies y > B/A (since z >= y, 2/y >= A/B => y <= 2B/A)
        # And y >= x
        start_y = max(x, B // A + 1)
        end_y = (2 * B) // A + 1
        
        # Limit the range to avoid excessive computation
        if end_y - start_y > 100000:
            end_y = start_y + 100000
        
        for y in range(start_y, end_y + 1):
            denom = A * y - B
            if denom <= 0:
                continue
            if (B * y) % denom != 0:
                continue
            z = (B * y) // denom
            if z <= y:
                continue
            if z > 10**12:
                continue
            # Check distinctness
            if x == y or x == z or y == z:
                continue
            return (x, y, z)
    return None

def _precompute():
    global _SOLUTIONS
    _SOLUTIONS = {}
    start_time = time.time()
    for n in range(3, 257):
        if time.time() - start_time > _DEADLINE:
            # If we're running out of time, try a faster fallback for remaining n
            _SOLUTIONS[n] = _find_solution_fast(n)
            continue
        _SOLUTIONS[n] = _find_solution(n)

def _find_solution_fast(n):
    """Fast fallback: try a limited set of x values."""
    start_x = n // 4 + 1
    end_x = min(start_x + 20, (3 * n) // 4 + 1)
    for x in range(start_x, end_x + 1):
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        start_y = max(x, B // A + 1)
        end_y = min((2 * B) // A + 1, start_y + 1000)
        for y in range(start_y, end_y + 1):
            denom = A * y - B
            if denom <= 0:
                continue
            if (B * y) % denom != 0:
                continue
            z = (B * y) // denom
            if z <= y or z > 10**12:
                continue
            if x == y or x == z or y == z:
                continue
            return (x, y, z)
    return None

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
