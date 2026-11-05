import math
import time

def construct():
    target = 123457
    t0 = time.time()
    deadline = t0 + 78

    # Build list of primes up to target using a linear sieve
    primes = []
    is_prime = bytearray(b'\x01') * (target + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, target + 1):
        if is_prime[i]:
            primes.append(i)
            if i * i <= target:
                step = i
                start = i * i
                is_prime[start:target+1:step] = b'\x00' * len(range(start, target+1, step))

    # Binary search on threshold K: find smallest prime K such that
    # there exist K-smooth a, b with a + b = target.
    # "K-smooth" means all prime factors <= K.
    #
    # For a given K, we build the set of all K-smooth numbers up to target,
    # then check if any pair sums to target.
    #
    # To make this fast: for small K, the set of K-smooth numbers is small.
    # We can enumerate them via DFS over primes <= K.

    def smooth_numbers_up_to(K, limit):
        """Return a set of all K-smooth numbers in [1, limit]."""
        # Collect primes <= K
        small_primes = [p for p in primes if p <= K]
        result = set()
        
        def dfs(idx, current):
            if current > limit:
                return
            if current >= 1:
                result.add(current)
            if idx >= len(small_primes):
                return
            p = small_primes[idx]
            # Multiply current by p repeatedly
            val = current
            while val * p <= limit:
                val *= p
                dfs(idx + 1, val)
        
        dfs(0, 1)
        return result

    # Binary search on the index into primes
    lo, hi = 0, len(primes) - 1
    best_a, best_b = 1, target - 1
    best_K = None

    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        K = primes[mid]
        
        # Get all K-smooth numbers up to target
        smooth = smooth_numbers_up_to(K, target)
        
        # Check if any a in smooth has target - a also in smooth
        found = False
        for a in sorted(smooth):
            if time.time() > deadline:
                break
            b = target - a
            if a < 1 or b < 1:
                continue
            if b in smooth:
                # Verify: both are K-smooth and sum to target
                found = True
                best_a, best_b = a, b
                best_K = K
                break
        
        if found:
            hi = mid - 1
        else:
            lo = mid + 1

    if best_K is not None:
        return [best_a, best_b]

    # Fallback: linear scan with LPF (shouldn't reach here for this target)
    lpf = [0] * (target + 1)
    lpf[1] = 1
    for i in range(2, target + 1):
        if lpf[i] == 0:
            for j in range(i, target + 1, i):
                lpf[j] = i

    best_score = max(lpf[1], lpf[target - 1])
    best_a, best_b = 1, target - 1
    half = target // 2
    for a in range(2, half + 1):
        if time.time() > deadline:
            break
        b = target - a
        s = max(lpf[a], lpf[b])
        if s < best_score:
            best_score = s
            best_a, best_b = a, b

    return [best_a, best_b]
