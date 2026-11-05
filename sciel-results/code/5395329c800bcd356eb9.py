"""Smooth two-summand decomposition of a prescribed integer."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    N = 123457
    # Sieve to compute largest prime factor for each number up to N
    # O(N log log N)
    lp = [0] * (N + 1)  # largest prime factor
    lp[1] = 1
    for i in range(2, N + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                if lp[j] == 0:
                    lp[j] = i
                else:
                    # i is a larger prime factor than current lp[j]
                    lp[j] = i

    # Now lp[x] holds the largest prime factor of x for x in [1, N]
    # We need to find a in [1, N-1] minimizing max(lp[a], lp[N-a])
    best_a = 1
    best_b = N - 1
    best_val = max(lp[1], lp[N - 1])

    for a in range(1, N):
        b = N - a
        val = max(lp[a], lp[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            if best_val == 1:
                # Can't do better than 1 (only if both are 1, impossible for sum > 2)
                break

    # Check time; if we're close to deadline, return immediately
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 5:
        pass  # We already have our answer

    return [best_a, best_b]
# EVOLVE-BLOCK-END
