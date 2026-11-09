import time
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    def phi(x):
        result, d, y = x, 2, x
        while d * d <= y:
            if y % d == 0:
                result -= result // d
                while y % d == 0:
                    y //= d
            d += 1
        return result - result // y if y > 1 else result

    # First, do a quick sieve-based scan up to 10^6 to find good candidates
    # Sieve phi up to 10^6
    N = 1000000
    phi_arr = list(range(N + 1))
    for i in range(2, N + 1):
        if phi_arr[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi_arr[j] -= phi_arr[j] // i

    # Find longest run of distinct totients in [1, N]
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

    # Now use remaining time to do random sampling at larger scales
    # to find potentially longer runs
    remaining = deadline - (time.time() - start_time)
    if remaining > 2:
        # Sample random starting points at various scales
        import random
        random.seed(42)
        # Try ranges around multiples of small LCMs or just random large starts
        for _ in range(200):
            if time.time() - start_time > deadline - 1:
                break
            # Choose a random start point
            # Focus on ranges where totients are likely to be distinct
            # e.g., near primes, or in sparse regions
            scale = random.choice([10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000])
            s = random.randint(1, max(1, 10**12 - 200))
            
            # Check a window of up to 200 values
            seen = set()
            k = 0
            for i in range(200):
                val = s + i
                if val > 10**12:
                    break
                pv = phi(val)
                if pv in seen:
                    break
                seen.add(pv)
                k += 1
            
            if k > best_k:
                best_k = k
                best_s = s

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
