"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    TARGET = 123457

    # Sieve of Eratosthenes to get all primes up to TARGET
    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                step = i
                start = i * i
                is_prime[start::step] = bytearray(len(is_prime[start::step]))
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(TARGET)

    # Incremental smooth number generation
    # smooths contains all numbers whose prime factors are among primes processed so far
    smooths = {1}
    
    for idx, q in enumerate(primes):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate new smooth numbers that include prime q
        # For each existing smooth number s, multiply by q, q^2, q^3, ...
        new_smooths = []
        # Convert to list to avoid modification during iteration issues,
        # but we only read from smooths, so we can iterate directly if we're careful.
        # However, since smooths is a set, iterating while adding is unsafe.
        # We only add to new_smooths, so iterating over smooths is safe.
        for s in smooths:
            val = s
            while val * q <= TARGET:
                val *= q
                new_smooths.append(val)
        
        # Add new smooths to the set
        for ns in new_smooths:
            smooths.add(ns)
        
        # Check for pair: only need to check newly added smooths
        # because old smooths were already checked in previous iterations
        # If a is new, and b = TARGET - a is in smooths (which includes old and new),
        # then max(P(a), P(b)) <= q, since all factors of a and b are <= q.
        for a in new_smooths:
            b = TARGET - a
            if b > 0 and b in smooths:
                return [a, b]
    
    # Fallback if no pair found within deadline (should not happen for this TARGET)
    # Use the current program's fallback logic if needed, but the loop should find it.
    # The current program's linear scan is a safe fallback but slower.
    # However, since we have spent time in the smooth generation, we might not have time for a full scan.
    # But for N=123457, the optimal K is likely small enough that we find it quickly.
    # If we didn't find it, we return a valid but potentially suboptimal pair.
    # Let's do a quick check of the current best known or just return [1, TARGET-1]
    # Actually, the linear scan in the original program is O(N) and fast enough if we have time.
    # But if we hit the deadline, we just return the best we can.
    # For robustness, let's just return [1, TARGET-1] as a valid pair.
    return [1, TARGET - 1]
# EVOLVE-BLOCK-END
