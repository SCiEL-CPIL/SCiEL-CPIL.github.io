"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill

    N = 512
    TARGET = 48

    def try_greedy(order, deadline):
        """Greedy: add x from order if valid. Returns selected list."""
        selected = []
        # dp[k]: bitset, bit s set iff s is sum of exactly k elements from selected
        dp = [0]  # dp[0] = 1 (sum 0 with 0 elements)
        dp[0] = 1
        # Precompute for speed: we'll track max_k = len(selected)

        for x in order:
            if time.time() > deadline:
                break
            n = len(selected)
            if n >= TARGET:
                break

            # Check 1: is x the mean of any subset of selected of size >= 2?
            # Need sum = k*x for some k in 2..n, achievable with k elements.
            valid = True
            for k in range(2, n + 1):
                target = k * x
                if target > (k * N):  # can't exceed k*max_val
                    break
                if (dp[k] >> target) & 1:
                    valid = False
                    break
            if not valid:
                continue

            # Check 2: for each a in selected, is a the mean of a subset of
            # (selected \ {a}) U {x} of size >= 2 that includes x?
            # Subset U of selected\{a}, size k >= 1, sum(U) = (k+1)*a - x
            for a in selected:
                # k ranges 1..n-1 (U subset of S\{a}, size up to n-1)
                for k in range(1, n):
                    target = (k + 1) * a - x
                    if target < 0:
                        continue
                    if target > (k * N):
                        break
                    if (dp[k] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            # Valid: add x
            selected.append(x)
            # Update dp: new_dp[k] = dp[k] | (dp[k-1] << x)
            new_dp = [0] * (n + 2)
            new_dp[0] = 1
            for k in range(1, n + 2):
                val = dp[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (dp[k - 1] << x)
                new_dp[k] = val
            dp = new_dp

        return selected

    # Try deterministic orderings first
    best = []
    for order in [range(1, N + 1), range(N, 0, -1)]:
        res = try_greedy(order, deadline)
        if len(res) > len(best):
            best = res
        if len(best) >= TARGET:
            return best

    # Randomized restarts with various orderings
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline - 2:
            break
        if len(best) >= TARGET:
            break

        # Vary the ordering strategy
        mode = trial % 5
        if mode == 0:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif mode == 1:
            order = list(range(N, 0, -1))
            rng.shuffle(order)
        elif mode == 2:
            # Sort by value mod a small prime (to group by residue)
            p = 7 + (trial % 10)
            order = sorted(range(1, N + 1), key=lambda v: v % p)
            rng.shuffle(order)  # still randomize within
        elif mode == 3:
            # Random priority
            order = list(range(1, N + 1))
            rng.shuffle(order)
        else:
            # Start with a random subset bias
            order = list(range(1, N + 1))
            # Bias toward mid-range values
            order.sort(key=lambda v: abs(v - N // 2))
            rng.shuffle(order)

        res = try_greedy(order, deadline)
        if len(res) > len(best):
            best = res

    return best

# EVOLVE-BLOCK-END
