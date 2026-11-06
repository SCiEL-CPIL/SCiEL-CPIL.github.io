"""Find the smallest n in [1,1000000] such that phi(n) is divisible by 120."""

def construct():
    import math
    import time
    
    deadline = time.time() + 85  # 85 second deadline, 5s safety margin
    N = 1000000
    
    # We need phi(n) % 120 == 0
    # 120 = 2^3 * 3 * 5
    # phi(n) = n * prod_{p|n} (1 - 1/p)
    
    # Strategy: iterate n from 1 upward, compute phi(n), check divisibility by 120.
    # For small n this is fast. We can use a sieve-like approach for efficiency.
    
    # Let's use a linear sieve or Euler's totient sieve to compute phi for all n up to N,
    # then find the smallest n with phi[n] % 120 == 0.
    
    # But actually, we can just iterate and compute phi on the fly for small n.
    # The answer should be quite small.
    
    def phi(n):
        """Compute Euler's totient of n."""
        if n == 1:
            return 1
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result
    
    # Search from 1 upward
    for n in range(1, N + 1):
        if time.time() > deadline:
            # Return best known answer (shouldn't happen for small n)
            break
        if phi(n) % 120 == 0:
            return n
    
    # Fallback: use a sieve approach
    phi_arr = list(range(N + 1))
    for i in range(2, N + 1):
        if phi_arr[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi_arr[j] -= phi_arr[j] // i
    
    for n in range(1, N + 1):
        if phi_arr[n] % 120 == 0:
            return n
    
    return -1
