import time

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    def phi(n):
        """Calculate Euler's totient of n."""
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    # We search for the smallest n, so we iterate from 1 upwards.
    # The solution is expected to be small, so this loop will terminate quickly.
    for n in range(1, 1000001):
        if phi(n) % 120 == 0:
            return n
        # Check time to ensure we don't exceed limits, though unlikely to be needed
        if time.time() > deadline:
            # If we somehow haven't found it (which is mathematically impossible for this range 
            # within the time limit if the algorithm is correct), we would return a fallback.
            # But phi(120) = 32, phi(121)=110, ... 
            # Actually, let's just let it run. The math guarantees a solution.
            # phi(300) = 80. phi(312) = 96. phi(320) = 128. 
            # Let's check a known small one: 
            # phi(120) = 32. 
            # phi(1) = 1
            # phi(2) = 1
            # ...
            # The first n with phi(n) divisible by 120 is likely around 300-400.
            # Returning n=1 as a fallback is invalid, but we won't reach here.
            pass

    # Fallback if no solution found (should not happen)
    return 1
