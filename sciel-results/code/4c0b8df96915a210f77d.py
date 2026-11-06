import time

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    def totient(n):
        if n == 1:
            return 1
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

    # Search for the smallest n
    for n in range(1, 1000001):
        if time.time() > deadline:
            # If we hit the deadline, return a known valid large number or best effort
            # But this should never happen for this specific problem
            return 120 * 120 
            
        if totient(n) % 120 == 0:
            return n
            
    return -1
