"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    """
    For each consecutive prime pair (lo, hi), find an integer m strictly between them
    whose least prime factor (LPF) is at least the prime gap (hi - lo).
    Returns a list of 32 entries, each an int witness or None.
    """
    deadline = time.time() + 80.0

    max_val = max(hi for _, hi in P['gaps'])
    
    # Precompute least prime factor via sieve
    lpf = list(range(max_val + 1))
    if max_val >= 0:
        lpf[0] = 0
    if max_val >= 1:
        lpf[1] = 1
    for i in range(2, int(max_val**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i*i, max_val + 1, i):
                if lpf[j] == j:
                    lpf[j] = i

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        found = False
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = int(m)
                found = True
                break
        # If not found via LPF, try direct check as fallback
        if not found:
            for m in range(lo + 1, hi):
                # Check if m has no prime factor < gap
                valid = True
                d = 2
                while d * d <= m and d < gap:
                    if m % d == 0:
                        valid = False
                        break
                    d += 1
                if valid:
                    result[i] = int(m)
                    found = True
                    break

    return result
# EVOLVE-BLOCK-END
