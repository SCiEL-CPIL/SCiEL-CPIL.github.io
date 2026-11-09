import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin

    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p : limit+1 : p] = False

    primes = np.where(is_prime)[0]
    N = len(primes)

    if N == 0:
        return []

    # Greedy: try from each starting prime, pick smallest valid gap
    best_seq = []
    for start_idx in range(N):
        if time.time() - start_time > DEADLINE - 5:
            break
        seq = [int(primes[start_idx])]
        last_gap = 0
        cur_idx = start_idx
        while True:
            next_idx = -1
            next_gap = float('inf')
            for j in range(cur_idx + 1, N):
                gap = int(primes[j] - primes[cur_idx])
                if gap >= last_gap:
                    next_gap = gap
                    next_idx = j
                    break
            if next_idx == -1:
                break
            seq.append(int(primes[next_idx]))
            last_gap = next_gap
            cur_idx = next_idx
        if len(seq) > len(best_seq):
            best_seq = seq

    # DP with proper parent tracking
    # dp_len[i][g] = max length of valid seq ending at primes[i] with last gap exactly g
    # best_prefix[i][g] = max length ending at i with last gap <= g
    # arg_gap[i][g] = the gap g' <= g that achieves best_prefix[i][g]
    # par_idx[i][g] = index j of previous prime in optimal seq for state (i, g)
    # par_gap[i][g] = gap g' at j that was used

    dp_len = np.zeros((N, limit + 1), dtype=np.uint16)
    best_prefix = np.zeros((N, limit + 1), dtype=np.uint16)
    arg_gap = np.zeros((N, limit + 1), dtype=np.uint16)
    par_idx = np.full((N, limit + 1), -1, dtype=np.int32)
    par_gap = np.zeros((N, limit + 1), dtype=np.uint16)

    # Init: length 1 sequences
    for i in range(N):
        dp_len[i, 0] = 1
        best_prefix[i, 0] = 1
        arg_gap[i, 0] = 0

    global_max = 1
    best_i = 0
    best_g = 0

    for i in range(1, N):
        if time.time() - start_time > DEADLINE - 10:
            break
        pi = int(primes[i])
        for j in range(i):
            d = pi - int(primes[j])
            if d > limit:
                continue
            val = best_prefix[j, d]
            if val > 0:
                new_len = val + 1
                if new_len > dp_len[i, d]:
                    dp_len[i, d] = new_len
                    par_idx[i, d] = j
                    par_gap[i, d] = arg_gap[j, d]

        # Build prefix structures for row i
        run_max = 0
        run_arg = 0
        for g in range(1, limit + 1):
            if dp_len[i, g] > run_max:
                run_max = dp_len[i, g]
                run_arg = g
            best_prefix[i, g] = run_max
            arg_gap[i, g] = run_arg
            if run_max > global_max:
                global_max = run_max
                best_i = i
                best_g = run_arg

    # Reconstruct
    if global_max <= len(best_seq):
        return best_seq

    seq = []
    cur_i = best_i
    cur_g = best_g
    while cur_i >= 0:
        seq.append(int(primes[cur_i]))
        if par_idx[cur_i, cur_g] == -1:
            break
        next_g = par_gap[cur_i, cur_g]
        cur_i = par_idx[cur_i, cur_g]
        cur_g = next_g

    seq.reverse()
    return seq
