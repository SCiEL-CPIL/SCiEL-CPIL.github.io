import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n_primes = len(primes)

    # dp[j] is a list of tuples (gap, length, prev_index, prev_gap)
    # representing the Pareto front of valid sequences ending at primes[j].
    dp = []
    for j in range(n_primes):
        dp.append([(0, 1, -1, -1)])

    best_len = 1
    best_index = 0

    for j in range(1, n_primes):
        if time.time() > deadline:
            break

        p_j = primes[j]
        current_pairs = [(0, 1, -1, -1)]

        for i in range(j):
            p_i = primes[i]
            gap_ij = p_j - p_i

            # Iterate over a copy of dp[i] in case we modify current_pairs
            # but dp[i] is not modified.
            for gap_i, len_i, prev_i, prev_gap_i in dp[i]:
                if gap_ij >= gap_i:
                    new_len = len_i + 1
                    new_pair = (gap_ij, new_len, i, gap_i)

                    # Check if new_pair is dominated by any in current_pairs
                    dominated = False
                    for g, l, pi, gi in current_pairs:
                        if g <= gap_ij and l >= new_len:
                            dominated = True
                            break
                    if dominated:
                        continue

                    # Remove pairs in current_pairs dominated by new_pair
                    filtered_pairs = []
                    for g, l, pi, gi in current_pairs:
                        if not (gap_ij <= g and new_len >= l):
                            filtered_pairs.append((g, l, pi, gi))
                    filtered_pairs.append(new_pair)
                    current_pairs = filtered_pairs

                    if new_len > best_len:
                        best_len = new_len
                        best_index = j

        dp[j] = current_pairs

    # Reconstruct the path
    path = []
    curr_idx = best_index
    curr_len = best_len

    # Find the entry in dp[best_index] with length best_len
    target_entry_idx = -1
    for k, (g, l, pi, gi) in enumerate(dp[best_index]):
        if l == best_len:
            target_entry_idx = k
            break

    if target_entry_idx == -1:
        return primes

    path.append(primes[curr_idx])

    while curr_len > 1:
        g, l, pi, gi = dp[curr_idx][target_entry_idx]
        path.append(primes[pi])

        # Find the corresponding entry in dp[pi]
        # It should have gap = gi and length = l - 1
        next_target_entry_idx = -1
        for k, (g2, l2, pi2, gi2) in enumerate(dp[pi]):
            if g2 == gi and l2 == l - 1:
                next_target_entry_idx = k
                break

        if next_target_entry_idx == -1:
            # This should not happen if logic is correct
            break

        curr_idx = pi
        curr_len = l - 1
        target_entry_idx = next_target_entry_idx

    path.reverse()
    return path
