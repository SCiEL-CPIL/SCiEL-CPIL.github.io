import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    def is_valid_addition(x, S):
        """Check if x can be added to set S without violating constraints."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                a = S_list[i]
                b = S_list[j]
                # Check x | a*b
                if (a * b) % x == 0:
                    return False
                # Check a | x*b
                if (x * b) % a == 0:
                    return False
                # Check b | x*a
                if (x * a) % b == 0:
                    return False
        return True
    
    def greedy_build(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S):
                S.add(x)
        return S
    
    # Strategy 1: Descending order (largest first)
    S1 = greedy_build(sorted(candidates, reverse=True))
    
    if time.time() > deadline:
        return sorted(S1)
    
    # Strategy 2: Primes first, then composites descending
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    order2 = sorted(composites, reverse=True) + primes
    S2 = greedy_build(order2)
    
    if time.time() > deadline:
        return sorted(max(S1, S2, key=len))
    
    # Strategy 3: Numbers with few divisors first (roughly primes and semiprimes)
    def num_divisors(n):
        count = 0
        for d in range(1, int(math.isqrt(n)) + 1):
            if n % d == 0:
                count += 1
                if d != n // d:
                    count += 1
        return count
    
    order3 = sorted(candidates, key=lambda x: (num_divisors(x), -x))
    S3 = greedy_build(order3)
    
    if time.time() > deadline:
        return sorted(max(S1, S2, S3, key=len))
    
    # Strategy 4: Start with large numbers, fill in gaps
    # Try multiple random-ish orderings
    import random
    random.seed(42)
    
    best_S = max(S1, S2, S3, key=len)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        # Mix: large numbers first with some randomness
        order = list(candidates)
        # Weight toward larger numbers
        random.shuffle(order)
        order.sort(key=lambda x: -x + random.randint(-20, 20))
        S_trial = greedy_build(order)
        if len(S_trial) > len(best_S):
            best_S = S_trial
    
    # Try to improve: add back removed numbers
    if time.time() < deadline - 5:
        removed = set(candidates) - best_S
        for x in sorted(removed, reverse=True):
            if time.time() > deadline:
                break
            if is_valid_addition(x, best_S):
                best_S.add(x)
    
    return sorted(best_S)
