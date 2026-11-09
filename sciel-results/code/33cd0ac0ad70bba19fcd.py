import random
import time

START_TIME = time.time()
DEADLINE = 58.0  # safety margin before the 90s kill (and well under the 60s eval budget)

def _remaining():
    return DEADLINE - (time.time() - START_TIME)

def check_validity(A):
    """Fast incremental check using bitmasks: no subset sum is reachable at two different cardinalities.

    Returns True iff for every integer s, the set of cardinalities k for which
    s is a sum of exactly k elements of A has size at most 1.
    """
    n = len(A)
    if n == 0:
        return True
    
    # dp[k] is a bitmask where bit i is 1 if sum i is achievable with k elements
    # Max sum is 256 * 60 = 15360, so we need ~15361 bits.
    # Python ints are arbitrary precision, so this is fine.
    
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements
    
    for a in A:
        # Update in reverse to avoid using the same element multiple times
        for k in range(n - 1, -1, -1):
            if dp[k] != 0:
                dp[k+1] |= (dp[k] << a)
        
        # Early termination check: if any sum appears in multiple cardinalities, it's invalid.
        # We can check this incrementally, but checking at the end of each addition is O(N^2) bitwise ops.
        # Since N is small (<=60), this is fast.
        # A sum s is invalid if it is in dp[i] and dp[j] for i != j.
        # Let's just check all pairs or use a cumulative mask.
        # Efficient check:
        # If we maintain a 'seen' mask, we can't just OR it because we need to detect overlap.
        # Overlap exists if (dp[i] & dp[j]) != 0 for any i != j.
        # We can do this by iterating k and checking against a 'used' mask?
        # No, 'used' would accumulate. We need to know if a bit was set in a *previous* k.
        # Let's just do the pairwise check or a smarter accumulation.
        
        # Optimization: 
        # If we process k from 0 to n, we can keep a `prev_union` mask.
        # If `dp[k] & prev_union` is not 0, then some sum in dp[k] was already seen in a lower cardinality.
        # Wait, dp[k] is updated. The new sums are in dp[k+1].
        # The conflict is if a sum s is in dp[i] and dp[j].
        
        # Let's just do the full check at the end of the loop for simplicity, as N is small.
        # Actually, we can check incrementally:
        # After updating dp, check if any dp[k] intersects with any dp[j] (j != k).
        # This is O(N^2) bitwise ANDs. 60*60 = 3600 ops. Very fast.
        
        # Let's optimize the check:
        # We can maintain a list of masks.
        # For each new a, we update dp.
        # Then we check validity.
        
        # Actually, we can just return False if a conflict is detected.
        # To detect conflict:
        # Let U = union of all dp[k].
        # If sum(bit_count(dp[k]) for k) > bit_count(U), then there is a duplicate.
        # This is O(N) bitwise ops.
        
        # Let's use this method:
        total_bits = 0
        union_mask = 0
        for k in range(n + 1):
            total_bits += bin(dp[k]).count('1')
            union_mask |= dp[k]
        if total_bits > bin(union_mask).count('1'):
            return False
            
    return True

def _try_add(current, current_set, candidates):
    """Greedily add the best (smallest) candidate that keeps validity. Returns (added, new_set)."""
    for c in candidates:
        if c in current_set:
            continue
        if check_validity(current + [c]):
            current_set.add(c)
            return True, c
    return False, None

def construct():
    if _remaining() < 1.0:
        return [1, 2, 4, 8, 16, 32, 64, 128]

    # Known valid base (powers of 2 are cardinality-separated).
    base = [1, 2, 4, 8, 16, 32, 64, 128]
    best = list(base)
    best_set = set(base)

    # Primary greedy build: try to add small elements first (small values keep
    # subset sums spread out and reduce cross-cardinality collisions).
    if _remaining() > 2.0:
        current = list(base)
        current_set = set(base)
        # candidate order: small first, then large (small is the heuristic sweet spot)
        candidates = [x for x in range(1, 257) if x not in current_set]
        candidates.sort()
        while _remaining() > 1.5:
            added, c = _try_add(current, current_set, candidates)
            if not added:
                break
            current.append(c)
        if len(current) > len(best):
            best = list(current)
            best_set = set(current)

    # Second pass with the remaining candidates (larger values) to squeeze more.
    if _remaining() > 2.0:
        current = list(best)
        current_set = set(best)
        remaining = [x for x in range(1, 257) if x not in current_set]
        remaining.sort()
        while _remaining() > 1.5:
            added, c = _try_add(current, current_set, remaining)
            if not added:
                break
            current.append(c)
        if len(current) > len(best):
            best = list(current)
            best_set = set(current)

    # Random-restart phase: escape local optima by starting from a random valid
    # seed of the current best size and greedily growing.
    if _remaining() > 3.0:
        size = len(best)
        rng = random.Random(12345)
        # Increase restarts since check is faster
        for _ in range(500): 
            if _remaining() < 1.5:
                break
            seed = sorted(rng.sample(range(1, 257), size))
            if not check_validity(seed):
                continue
            cur = list(seed)
            cur_set = set(seed)
            cands = [x for x in range(1, 257) if x not in cur_set]
            cands.sort()
            while _remaining() > 1.2:
                added, c = _try_add(cur, cur_set, cands)
                if not added:
                    break
                cur.append(c)
            if len(cur) > len(best):
                best = list(cur)
                best_set = set(cur)

    result = [int(x) for x in best]
    return result
