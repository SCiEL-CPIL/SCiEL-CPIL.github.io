import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    LIMIT = 10**12
    answers = [None] * 254
    
    def get_divisors(n):
        """Get all positive divisors of n."""
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
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # x must satisfy 1/x < 4/n, so x > n/4
        x_min = n // 4 + 1
        for x in range(x_min, min(x_min + 10000, LIMIT + 1)):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Get divisors of b_sq
            divs = get_divisors(b_sq)
            
            for d in divs:
                # y = (d + b) / a
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y <= 0 or y <= x:
                    continue
                if y > LIMIT:
                    continue
                
                # z = (b_sq // d + b) / a
                e = b_sq // d
                if (e + b) % a != 0:
                    continue
                z = (e + b) // a
                if z <= 0 or z <= y:
                    continue
                if z > LIMIT:
                    continue
                
                # Check distinctness
                if x != y and y != z and x != z:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
