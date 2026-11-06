import time
from math import lcm
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)

    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = [L // d for d in denominators]

    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    n1 = mid
    n2 = n - mid

    def check_time():
        return time.time() > deadline

    def dfs_best_subsets(weights_list, target, limit=2000000):
        """
        Find subset sums < target using DFS with suffix-sum pruning.
        Returns list of (sum, mask) sorted by sum descending (best first).
        """
        m = len(weights_list)
        # Sort by weight descending for better pruning
        order = sorted(range(m), key=lambda i: -weights_list[i])
        sorted_w = [weights_list[i] for i in order]

        # Suffix sums: suffix[i] = sum of sorted_w[i:]
        suffix = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix[i] = suffix[i + 1] + sorted_w[i]

        results = []
        count = 0

        def dfs(idx, current_sum, mask):
            nonlocal count
            if check_time():
                return
            if count >= limit:
                return
            if idx == m:
                if current_sum < target:
                    results.append((current_sum, mask))
                    count += 1
                return

            # Pruning: if even adding all remaining can't get closer to target,
            # and current_sum is already good, we can still record it
            # But more importantly: if current_sum + suffix[idx] <= best_found, skip
            # Actually for our use case we want ALL sums < target, but we can prune
            # branches where current_sum + suffix[idx] < some threshold

            # Option: include item idx
            new_sum = current_sum + sorted_w[idx]
            if new_sum < target:
                dfs(idx + 1, new_sum, mask | (1 << idx))

            # Option: exclude item idx
            # Pruning: if current_sum + suffix[idx] <= 0, no point (but weights are positive)
            # Better pruning: if we already have many results and current_sum + suffix[idx]
            # is much less than target, we might skip, but we want the BEST (closest to target)
            # So we should explore branches that can reach close to target first

            dfs(idx + 1, current_sum, mask)

        dfs(0, 0, 0)

        # Convert masks back: mask bit i corresponds to sorted position i
        # We need to map back to original indices
        final_results = []
        for s, mask in results:
            orig_mask = 0
            for i in range(m):
                if mask & (1 << i):
                    orig_mask |= (1 << order[i])
            final_results.append((s, orig_mask))

        # Sort by sum descending (closest to target first)
        final_results.sort(key=lambda x: -x[0])
        return final_results

    # Strategy: For each half, find the best subsets (closest to L but < L)
    # Then combine them

    print("Finding best subsets for half 1...")
    # For half 1, we want sums < L (since half 2 can contribute 0)
    # But actually, we want to combine, so let's find top candidates for each half
    # that are close to L/2 or whatever allows good combination

    # Better approach: find top K subsets from each half, then try all pairs
    # But K can be large. Let's use the DFS to find subsets close to specific targets.

    # Alternative: Use DFS to find the single best subset for half 1 given a target,
    # and similarly for half 2, then iterate.

    # Actually, the cleanest approach:
    # 1. Generate a good set of candidate sums for half 2 (sorted)
    # 2. For each candidate sum from half 1, binary search in half 2

    # Let's generate candidates for half 2 first (smaller half might be faster)
    print("Generating candidates for half 2...")
    # We want sums < L for half 2
    candidates2 = dfs_best_subsets(w2, L, limit=500000)
    print(f"  Got {len(candidates2)} candidates for half 2")

    if check_time():
        # Fallback
        return [0]

    # Sort candidates2 by sum for binary search
    candidates2.sort(key=lambda x: x[0])
    c2_vals = [s for s, _ in candidates2]

    print("Generating candidates for half 1...")
    candidates1 = dfs_best_subsets(w1, L, limit=500000)
    print(f"  Got {len(candidates1)} candidates for half 1")

    if check_time():
        # Use what we have
        pass

    best_sum = -1
    best_m1 = 0
    best_m2 = 0

    # For each candidate in half 1, find best in half 2
    for s1, m1 in candidates1:
        if check_time():
            break

        target = L - 1 - s1
        if target < 0:
            continue

        idx = bisect.bisect_right(c2_vals, target)
        if idx > 0:
            s2, m2 = candidates2[idx - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_m1 = m1
                best_m2 = m2

    # Reconstruct indices
    if best_sum == -1:
        best_indices = [0]
    else:
        best_indices = []
        for i in range(mid):
            if best_m1 & (1 << i):
                best_indices.append(i)
        for i in range(mid, n):
            if best_m2 & (1 << (i - mid)):
                best_indices.append(i)

    if not best_indices:
        best_indices = [0]

    # Verify
    s = sum(weights[i] for i in best_indices)
    if s >= L:
        best_indices.pop()
        if not best_indices:
            best_indices = [0]

    print(f"Best sum: {best_sum}/{L}")
    print(f"Deficit: {L - best_sum}")
    print(f"Indices: {best_indices}")

    return best_indices
