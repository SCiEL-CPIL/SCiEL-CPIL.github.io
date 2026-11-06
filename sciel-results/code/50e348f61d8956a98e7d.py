# EVOLVE-BLOCK-START
def construct():
    import time
    from math import isqrt

    deadline = time.time() + 55.0

    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = [False] * (((N - start) // step) + 1)
    primes = [i for i, v in enumerate(is_prime) if v]
    num_primes = len(primes)

    # dp[i] = list of (gap, length, parent_j, parent_state_idx)
    dp = [[] for _ in range(num_primes)]

    for i in range(num_primes):
        if time.time() > deadline:
            break
        dp[i].append((0, 1, -1, -1))

        for j in range(i):
            gap = primes[i] - primes[j]
            # Iterate over states in dp[j]
            for psi, state in enumerate(dp[j]):
                if state[0] <= gap:
                    dp[i].append((gap, state[1] + 1, j, psi))

        # Prune Pareto-optimal states
        if dp[i]:
            # Find max length for each gap
            best_for_gap = {}
            for s in dp[i]:
                g = s[0]
                l = s[1]
                if g not in best_for_gap or l > best_for_gap[g][1]:
                    best_for_gap[g] = s
            
            # Sort by gap
            items = sorted(best_for_gap.values(), key=lambda x: x[0])
            
            # Keep frontier
            frontier = []
            max_len = 0
            for s in items:
                if s[1] > max_len:
                    frontier.append(s)
                    max_len = s[1]
            dp[i] = frontier

    # Find best
    best_len = 0
    best_i = -1
    best_state_idx = -1
    for i in range(num_primes):
        for si, s in enumerate(dp[i]):
            if s[1] > best_len:
                best_len = s[1]
                best_i = i
                best_state_idx = si

    if best_i == -1:
        return [2] if primes else []

    # Reconstruct
    chain = []
    i = best_i
    si = best_state_idx
    while i != -1:
        chain.append(primes[i])
        s = dp[i][si]
        if s[2] == -1:
            break
        i = s[2]
        si = s[3]

    chain.reverse()

    # Validate
    if len(chain) >= 2:
        gaps = [chain[k+1] - chain[k] for k in range(len(chain)-1)]
        valid = True
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                valid = False
                break
        if valid:
            return chain

    # Fallback
    values = []
    for p in primes:
        if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
            values.append(p)
    return values
# EVOLVE-BLOCK-END
