import time

def construct():
    MAX_DENOM = 10**12
    DEADLINE = time.time() + 85.0  # safety margin before 90s kill
    answers = []
    
    def get_divisors(num):
        """Get all divisors of num using factorization."""
        if num == 0:
            return [0]
        factors = {}
        d = 2
        temp = num
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 1)
        
        # Generate divisors from prime factorization
        divisors = [1]
        for prime, exp in factors.items():
            new_divs = []
            for d in divisors:
                for e in range(exp + 1):
                    new_divs.append(d * (prime ** e))
            divisors = new_divs
        return divisors
    
    for n in range(3, 257):
        if time.time() > DEADLINE:
            answers.append(None)
            continue
        
        found = None
        # Try various values of x starting from smallest possible
        # x must satisfy 4x > n (so a = 4x - n > 0)
        # and x < n (since 1/x > 4/n means... actually not necessarily)
        # Let's try x from ceil(n/4)+1 upward, but also try some other values
        
        for x in range(n // 4 + 1, n + 50):  # limited range for x
            if time.time() > DEADLINE:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Get divisors of b_sq
            divs = get_divisors(b_sq)
            
            for d in divs:
                if time.time() > DEADLINE:
                    break
                # y = (d + b) / a must be integer
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y <= 0 or y <= x:
                    continue
                # z = (b_sq // d + b) / a must be integer
                if b_sq % d != 0:
                    continue
                rem = b_sq // d
                if (rem + b) % a != 0:
                    continue
                z = (rem + b) // a
                if z <= 0 or z <= y:
                    continue
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                # Check distinctness: x, y, z all distinct
                if x != y and x != z and y != z:
                    # Verify the equation
                    if x * y * z * 4 == n * (y * z + x * z + x * y):
                        found = [x, y, z]
                        break
            if found:
                break
        
        answers.append(found)
    
    return answers
