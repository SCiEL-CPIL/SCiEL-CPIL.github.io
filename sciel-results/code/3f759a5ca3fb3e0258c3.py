"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Use sieve to compute phi values up to a large N
    # Start with a reasonable N and expand if time allows
    N = 5000000  # 5 million
    
    def compute_phi_sieve(n):
        """Compute Euler's totient function for all numbers up to n using sieve."""
        phi = list(range(n + 1))
        for i in range(2, n + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, n + 1, i):
                    phi[j] -= phi[j] // i
        return phi
    
    # Compute phi values
    phi_arr = compute_phi_sieve(N)
    
    # Find longest run of consecutive integers with distinct totients
    best_k = 1
    best_s = 1
    for s in range(1, N - 100):
        if time.time() - start_time > deadline - 5:
            break
        seen = set()
        k = 0
        while s + k <= N and phi_arr[s + k] not in seen:
            seen.add(phi_arr[s + k])
            k += 1
            if k > best_k:
                best_k = k
                best_s = s
            if k >= 200:  # Max allowed
                break
        # Optimization: if we've already found a very long run, we can skip ahead
        if best_k > 50 and s + k - 1 > s + 1000:
            # We can potentially skip forward
            pass
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
