"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s

    # Sieve primes up to 10000
    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)

    # DP with Pareto frontier:
    # dp[i] = list of (last_gap, length) pairs representing non-dominated states
    # for sequences ending at primes[i]. A state (g1, l1) dominates (g2, l2)
    # if g1 <= g2 and l1 >= l2 (smaller last gap is better for extension,
    # larger length is better).
    #
    # For each prime i, we consider all j < i as potential predecessors.
    # The gap from j to i is primes[i] - primes[j].
    # If gap >= last_gap of some state ending at j, we can extend that state.
    #
    # To speed up: for each i, we iterate j from i-1 down to 0.
    # The gap increases as j decreases, so once gap >= some threshold,
    # we can use bisect to find which states at j are extendable.
    # But the dominant cost is O(n^2 * avg_frontier_size).
    # With n ~ 1229, n^2 ~ 1.5M, and frontiers are small (typically < 10),
    # this should be fast enough.

    # dp[i] is a list of (gap, length) sorted by gap ascending.
    # We maintain the Pareto frontier: as gap increases, length must strictly increase.
    dp = [[] for _ in range(n)]
    parent = [[-1] * 100 for _ in range(n)]  # parent[i][k] = j that produced state k at i
    # Actually, let's store parent as a parallel list to dp[i]

    # Initialize: each prime alone is a sequence of length 1 with last_gap = 0
    for i in range(n):
        dp[i] = [(0, 1)]

    best_len = 1
    best_end = 0

    for i in range(1, n):
        if time.time() - t0 > DEADLINE:
            break

        new_states = []
        pi = primes[i]

        for j in range(i):
            gap = pi - primes[j]
            if gap <= 0:
                continue
            # For each state (g, l) in dp[j], if gap >= g, we can extend
            # to get state (gap, l+1) at position i
            for g, l in dp[j]:
                if gap >= g:
                    new_states.append((gap, l + 1, j))
                # Since dp[j] is sorted by gap ascending, and we need gap >= g,
                # once gap < g for a later entry (which has larger g), we can break
                # Actually dp[j] is sorted by gap ascending, so if gap < g for current,
                # all subsequent have larger g, so we can break.
                else:
                    # gap < g, and since g is increasing, all further states have larger g
                    break

        if not new_states:
            dp[i] = [(0, 1)]
            continue

        # Now we need to reduce new_states to a Pareto frontier.
        # Sort by gap ascending, and for equal gaps keep max length.
        # Then scan: keep only states where length is strictly greater than all previous.
        new_states.sort(key=lambda x: (x[0], -x[1]))

        # Deduplicate: for same gap, keep max length
        deduped = []
        prev_gap = None
        max_l_for_gap = 0
        for gap, l, j in new_states:
            if gap == prev_gap:
                if l > max_l_for_gap:
                    max_l_for_gap = l
            else:
                if prev_gap is not None:
                    deduped.append((prev_gap, max_l_for_gap))
                prev_gap = gap
                max_l_for_gap = l
        if prev_gap is not None:
            deduped.append((prev_gap, max_l_for_gap))

        # Now extract Pareto frontier: scan in order of increasing gap,
        # keep state if its length is strictly greater than max so far
        frontier = []
        max_l = 0
        for gap, l in deduped:
            if l > max_l:
                frontier.append((gap, l))
                max_l = l

        dp[i] = frontier

        # Track best
        if frontier:
            ml = frontier[-1][1]
            if ml > best_len:
                best_len = ml
                best_end = i

    # Reconstruct the best sequence
    # We need to trace back. The issue is that dp[i] doesn't store which j produced which state.
    # Let's redo with explicit parent tracking.

    # Reset
    dp2 = [[] for _ in range(n)]
    # parent[i] will be a list parallel to dp2[i], storing the j index
    parent2 = [[] for _ in range(n)]

    for i in range(n):
        dp2[i] = [(0, 1)]
        parent2[i] = [-1]  # no parent for length-1 sequence

    best_len2 = 1
    best_end2 = 0

    for i in range(1, n):
        if time.time() - t0 > DEADLINE:
            break

        new_states = []
        pi = primes[i]

        for j in range(i):
            gap = pi - primes[j]
            if gap <= 0:
                continue
            for k, (g, l) in enumerate(dp2[j]):
                if gap >= g:
                    new_states.append((gap, l + 1, j, k))
                else:
                    break

        if not new_states:
            dp2[i] = [(0, 1)]
            parent2[i] = [-1]
            continue

        new_states.sort(key=lambda x: (x[0], -x[1]))

        # Deduplicate
        deduped = []
        prev_gap = None
        best_l = 0
        best_jk = None
        for gap, l, j, k in new_states:
            if gap == prev_gap:
                if l > best_l:
                    best_l = l
                    best_jk = (j, k)
            else:
                if prev_gap is not None:
                    deduped.append((prev_gap, best_l, best_jk))
                prev_gap = gap
                best_l = l
                best_jk = (j, k)
        if prev_gap is not None:
            deduped.append((prev_gap, best_l, best_jk))

        # Pareto frontier
        frontier = []
        frontier_parents = []
        max_l = 0
        for gap, l, jk in deduped:
            if l > max_l:
                frontier.append((gap, l))
                frontier_parents.append(jk)
                max_l = l

        dp2[i] = frontier
        parent2[i] = frontier_parents

        if frontier:
            ml = frontier[-1][1]
            if ml > best_len2:
                best_len2 = ml
                best_end2 = i

    # Reconstruct
    seq = []
    cur_i = best_end2
    # Find the state index with max length at best_end2
    if dp2[cur_i]:
        # The last state in the frontier has the max length
        state_idx = len(dp2[cur_i]) - 1
    else:
        state_idx = 0

    while cur_i >= 0:
        seq.append(primes[cur_i])
        if state_idx >= 0 and state_idx < len(parent2[cur_i]):
            jk = parent2[cur_i][state_idx]
            if jk is None or jk == (-1, -1):
                break
            j, k = jk
            cur_i = j
            state_idx = k
        else:
            break

    seq.reverse()

    # Validate: check non-decreasing gaps
    if len(seq) >= 2:
        gaps_seq = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        valid = all(gaps_seq[i] <= gaps_seq[i+1] for i in range(len(gaps_seq)-1))
        if not valid:
            # Fallback to a simple valid sequence
            seq = [primes[0]]
            for p in primes[1:]:
                if len(seq) < 2 or p - seq[-1] >= seq[-1] - seq[-2]:
                    seq.append(p)

    return seq
# EVOLVE-BLOCK-END
