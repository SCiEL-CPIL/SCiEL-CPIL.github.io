"""Improved program: cardinality-indexed subset-sum DP with fast collision check."""
import time
import random

DEADLINE_SECONDS = 85.0  # safety margin before 90s kill
MAX_N = 60


def _try_add(dp, used_sums, n, x):
    """Try to add x. Returns (new_dp, new_used, success)."""
    new_dp = dp[:]
    for k in range(n + 1, 0, -1):
        shifted = dp[k - 1] << x
        new_dp[k] = dp[k] | shifted
    # Check for collisions: new sums in level k must not be in used_sums of other levels
    # Efficient: for each level, new bits = new_dp[k] & ~dp[k]
    # But we need to check against ALL other levels' full bitsets
    # Fast approach: compute union of all other levels' bits, check new bits against it
    # Even faster: check new bits against used_sums (which is OR of all levels)
    # But used_sums includes dp[k] itself, so we need to be careful.
    # Actually: new bits at level k are (dp[k-1] << x) & ~dp[k]
    # These new bits must not appear in any OTHER level.
    # other_sums = used_sums & ~dp[k]  (all sums used by other levels)
    # Check: new_bits & other_sums == 0 for all k
    
    for k in range(1, n + 2):
        new_bits = new_dp[k] & ~dp[k]  # bits newly added at level k
        if new_bits:
            other_sums = used_sums & ~dp[k]
            if new_bits & other_sums:
                return None, None, False
    new_used = used_sums
    for k in range(n + 2):
        new_used |= new_dp[k]
    return new_dp, new_used, True


def _greedy_build(candidates, deadline, seed=0):
    """Greedily build valid set. At each step, pick best candidate."""
    current = []
    dp = [0] * (MAX_N + 1)
    dp[0] = 1
    used_sums = 1
    n = 0
    remaining = list(candidates)
    
    while n < MAX_N and remaining:
        if time.time() > deadline:
            break
        
        # Find best candidate to add
        best_x = None
        best_new_dp = None
        best_new_used = None
        best_score = -1
        
        # Limit candidates to consider for speed
        check_limit = min(len(remaining), 80)
        for x in remaining[:check_limit]:
            new_dp, new_used, ok = _try_add(dp, used_sums, n, x)
            if ok:
                # Score: prefer candidates that add fewer new sums (less "spread")
                # or prefer larger x (more gap)
                # Let's score by: -new_total_sums (fewer new sums is better)
                new_total = 0
                for k in range(n + 2):
                    new_total += bin(new_dp[k]).count('1')
                score = -new_total + x * 0.1  # slight preference for larger x
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_dp = new_dp
                    best_new_used = new_used
        
        if best_x is None:
            # Try all remaining if none worked in first check_limit
            for x in remaining[check_limit:]:
                if time.time() > deadline:
                    break
                new_dp, new_used, ok = _try_add(dp, used_sums, n, x)
                if ok:
                    new_total = 0
                    for k in range(n + 2):
                        new_total += bin(new_dp[k]).count('1')
                    score = -new_total + x * 0.1
                    if score > best_score:
                        best_score = score
                        best_x = x
                        best_new_dp = new_dp
                        best_new_used = new_used
        
        if best_x is None:
            break
        
        dp = best_new_dp
        used_sums = best_new_used
        current.append(best_x)
        n += 1
        remaining.remove(best_x)
    
    return current


def _local_search_improve(s, deadline):
    """Try to improve set s by swapping elements."""
    if not s:
        return s
    
    best = s[:]
    best_size = len(best)
    
    # Build dp for best
    dp = [0] * (MAX_N + 1)
    dp[0] = 1
    used_sums = 1
    n = len(best)
    for val in best:
        for k in range(n, 0, -1):
            dp[k] = dp[k] | (dp[k - 1] << val)
        n += 1
    # Rebuild used_sums
    used_sums = 0
    for k in range(n + 1):
        used_sums |= dp[k]
    
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            # Try removing best[i] and adding a new element
            # First, build dp without best[i]
            removed = best[i]
            # Rebuild dp without element i
            dp_no_i = [0] * (MAX_N + 1)
            dp_no_i[0] = 1
            n_no_i = 0
            for j, val in enumerate(best):
                if j == i:
                    continue
                for k in range(n_no_i + 1, 0, -1):
                    dp_no_i[k] = dp_no_i[k] | (dp_no_i[k - 1] << val)
                n_no_i += 1
            
            used_no_i = 0
            for k in range(n_no_i + 1):
                used_no_i |= dp_no_i[k]
            
            # Try adding each candidate not in best
            for x in range(1, 257):
                if x in best:
                    continue
                new_dp, new_used, ok = _try_add(dp_no_i, used_no_i, n_no_i, x)
                if ok:
                    # Same size, but different composition
                    # Accept if it allows further growth
                    candidate = best[:]
                    candidate[i] = x
                    # Try to extend
                    dp_ext = new_dp
                    used_ext = new_used
                    n_ext = n_no_i + 1
                    for y in range(1, 257):
                        if y in candidate:
                            continue
                        if time.time() > deadline:
                            break
                        new_dp2, new_used2, ok2 = _try_add(dp_ext, used_ext, n_ext, y)
                        if ok2:
                            dp_ext = new_dp2
                            used_ext = new_used2
                            candidate.append(y)
                            n_ext += 1
                    if len(candidate) > best_size:
                        best = candidate
                        best_size = len(best)
                        improved = True
                        break
            if improved:
                break
    
    return best


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Descending
    candidates = list(range(256, 0, -1))
    current = _greedy_build(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 2: Ascending
    candidates = list(range(1, 257))
    current = _greedy_build(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 3: Spaced
    for step in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        candidates = list(range(step, 257, step))
        current = _greedy_build(candidates, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Strategy 4: Random restarts
    random.seed(42)
    for attempt in range(3000):
        if time.time() > deadline:
            break
        
        if attempt % 5 == 0:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif attempt % 5 == 1:
            candidates = list(range(256, 0, -1))
            for _ in range(10):
                i, j = random.randint(0, 255), random.randint(0, 255)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        elif attempt % 5 == 2:
            step = random.randint(2, 10)
            start = random.randint(1, step)
            candidates = list(range(start, 257, step))
            random.shuffle(candidates)
        elif attempt % 5 == 3:
            large = list(range(129, 257))
            small = list(range(1, 129))
            random.shuffle(small)
            candidates = large + small
        else:
            # Power-of-2 like
            candidates = []
            for e in range(8, -1, -1):
                base = 1 << e
                for d in range(0, min(base, 32)):
                    val = base + d
                    if 1 <= val <= 256:
                        candidates.append(val)
            random.shuffle(candidates)
        
        current = _greedy_build(candidates, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Local search on best
    if best_set and time.time() < deadline:
        improved = _local_search_improve(best_set, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    return list(best_set)
