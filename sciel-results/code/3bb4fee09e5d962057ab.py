def construct():
    candidates = list(range(2, 257))
    # Sort by number of divisors to prioritize "safe" numbers like primes
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
        
    candidates.sort(key=num_divisors)
    
    S = []
    for x in candidates:
        valid = True
        # Check against all pairs in S
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                a = S[i]
                b = S[j]
                # Triple (x, a, b)
                if (a * b) % x == 0:
                    valid = False
                    break
                # Triple (a, x, b)
                if (x * b) % a == 0:
                    valid = False
                    break
                # Triple (b, x, a)
                if (x * a) % b == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.append(x)
            
    return S
