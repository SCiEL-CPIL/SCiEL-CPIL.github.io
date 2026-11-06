import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s kill
    n = 512
    target = n  # Try to maximize as much as possible, no artificial cap of 48

    def add_to_dp(dp, x, m):
        # dp has length m+1 (0..m); returns new dp of length m+2 (0..m+1) for set ∪ {x}
        new_dp = dp + [0]
        for k in range(m + 1, 0, -1):
            new_dp[k] |= (new_dp[k - 1] << x)
        return new_dp

    def is_valid_candidate(C, dp, dp_without, x):
        m = len(C)
        # Check A: x is mean of a k-subset of C, k>=2
        for k in range(2, m + 1):
            target_sum = k * x
            if target_sum > m * n:
                continue
            if (dp[k] >> target_sum) & 1:
                return False
        # Check B: existing y becomes mean of a subset of C U {x} containing x
        for y in C:
            # dp_without[y] represents C \ {y}, size m-1, list length m (0..m-1)
            dpy = dp_without[y]
            for j in range(1, m):  # j = size of U subset of C\{y}
                target_sum = (j + 1) * y - x
                if target_sum < 0:
                    continue
                if target_sum > (m - 1) * n:
                    continue
                if (dpy[j] >> target_sum) & 1:
                    return False
        return True

    def full_validate(S):
        # Returns True if S is valid
        m = len(S)
        if m == 0:
            return True
        # Build dp for S
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in S:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        # For each x in S, check no subset of S\{x} of size k>=2 sums to k*x
        for x in S:
            # Build dp for S \ {x}
            dp2 = [0] * (m)  # S\{x} has m-1 elements, indices 0..m-1
            dp2[0] = 1
            for val in S:
                if val == x:
                    continue
                for k in range(m - 1, 0, -1):
                    dp2[k] |= (dp2[k - 1] << val)
            for k in range(2, m):
                target_sum = k * x
                if target_sum > (m - 1) * n:
                    continue
                if (dp2[k] >> target_sum) & 1:
                    return False
        return True

    def trim_to_valid(S):
        # Remove elements until valid
        S = list(S)
        while S and not full_validate(S):
            # remove the "most problematic" element: the one that is a mean
            removed = False
            for x in S:
                # check if x is a mean of a subset of S\{x}
                m = len(S)
                dp2 = [0] * (m)
                dp2[0] = 1
                for val in S:
                    if val == x:
                        continue
                    for k in range(m - 1, 0, -1):
                        dp2[k] |= (dp2[k - 1] << val)
                bad = False
                for k in range(2, m):
                    target_sum = k * x
                    if target_sum > (m - 1) * n:
                        continue
                    if (dp2[k] >> target_sum) & 1:
                        bad = True
                        break
                if bad:
                    S.remove(x)
                    removed = True
                    break
            if not removed:
                # no single element is a mean; should not happen if full_validate failed
                # remove last element as fallback
                S.pop()
        return S

    def try_swap(C, dp, dp_without, candidates):
        # Try to find a swap (y in C, z in candidates) such that C' = C - {y} + {z} is valid
        # and potentially allows further growth.
        # For simplicity, just check if C' is valid.
        # This is expensive, so we only do it if stuck.
        m = len(C)
        # We need to check if removing y and adding z keeps the set valid.
        # This requires checking means for all elements in C'.
        # A cheaper check: 
        # 1. z is not mean of any subset of C-{y}
        # 2. No y' in C-{y} is mean of any subset of C-{y} U {z}
        
        # Let's just brute force the swap validation for a few random pairs if stuck.
        # But we need dp for C-{y}. We have dp_without[y].
        # To check if z is valid in C-{y}, we use is_valid_candidate(C-{y}, dp_without[y], ..., z)
        # But we also need to check if existing elements in C-{y} are still valid with z added.
        # is_valid_candidate checks:
        # A: z is mean of subset of C-{y}
        # B: existing y' in C-{y} becomes mean of subset of (C-{y}) U {z}
        # This is exactly what we need!
        
        # So, for a candidate swap (y, z):
        # 1. Check if z is valid for set C-{y}.
        #    Use is_valid_candidate with C_minus_y = C - {y}, dp_minus_y = dp_without[y].
        #    Note: is_valid_candidate expects dp_without for the *new* set?
        #    No, is_valid_candidate(C, dp, dp_without, x) checks if x can be added to C.
        #    It uses dp (of C) and dp_without (of C\{y} for y in C).
        #    So if we want to add z to C-{y}, we need:
        #    - dp of C-{y} (which is dp_without[y])
        #    - dp_without for each element in C-{y} relative to C-{y}.
        #    We don't have dp_without for C-{y} elements directly.
        #    We have dp_without[y'] for y' in C, which is dp of C-{y'}.
        #    This is not the same as dp of (C-{y})-{y'}.
        
        # So is_valid_candidate cannot be directly reused for the swap target set without rebuilding dp_without.
        # Rebuilding dp_without is O(m^2).
        # Given the time budget, we might skip complex swaps and rely on more random restarts 
        # or a better initial ordering.
        
        return False

    best = []
    rng = random.Random(12345)
    max_trials = 50000 # Increase trials since we removed the early break at 48

    for trial in range(max_trials):
        if time.time() > deadline - 5.0:
            break
        # If we found a very large set, we might stop early? 
        # No, we want to maximize.
        
        order = list(range(1, n + 1))
        rng.shuffle(order)

        C = []
        dp = [1]  # dp[0] = 1
        dp_without = {}

        for x in order:
            if time.time() > deadline - 10.0:
                break
            if is_valid_candidate(C, dp, dp_without, x):
                m = len(C)
                # add x
                # dp_without[x] = dp of C (before adding x)
                dp_without[x] = list(dp)
                # update dp_without[y] for y in C to include x
                for y in C:
                    dpy = dp_without[y] + [0]
                    for k in range(m, 0, -1):
                        dpy[k] |= (dpy[k - 1] << x)
                    dp_without[y] = dpy
                # update dp
                dp = add_to_dp(dp, x, m)
                C.append(x)
                
                # If we reach 48, we can continue trying to add more
                # No break here!

        if len(C) > len(best):
            best = list(C)
            # If we hit 50 or more, we are likely at the max, but keep going to be sure
            # Or if we hit 48, we might be stuck, but maybe 49 is possible?

    # Final safety: validate and trim
    if best and not full_validate(best):
        best = trim_to_valid(best)

    return best
