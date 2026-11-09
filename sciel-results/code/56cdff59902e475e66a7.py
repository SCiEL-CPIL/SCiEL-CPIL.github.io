import time
import random
import sys

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    sys.setrecursionlimit(10000)

    def is_valid(subset):
        reachable = 1
        for x in subset:
            new_reachable = reachable | (reachable << x)
            if (new_reachable >> 200) & 1:
                return False
            reachable = new_reachable
        return True

    def generate_bad_subsets():
        """
        Generates bitmasks of all subsets of [1, 100] that sum to 200.
        Returns None if the list is too large or takes too long.
        """
        # dp[s] is a list of bitmasks
        dp = [[] for _ in range(201)]
        dp[0].append(0)
        
        # Limit generation time
        gen_start = time.time()
        
        for i in range(1, 101):
            bit = 1 << i
            # Iterate downwards to avoid using i multiple times
            for s in range(200, i - 1, -1):
                if dp[s - i]:
                    # Extend dp[s] with new masks
                    # Optimization: if list gets too big, abort
                    if len(dp[s]) > 5000:
                        return None
                    dp[s].extend(m | bit for m in dp[s - i])
            
            if time.time() - gen_start > 5.0:
                return None
        
        return dp[200]

    def greedy_hitting_set(bad_subsets):
        """
        Finds a hitting set R for the given bad subsets.
        Returns the set of elements in R.
        """
        n_bad = len(bad_subsets)
        if n_bad == 0:
            return set()
        
        # cover_mask[i] is a bitmask of indices of bad subsets containing element i
        # We use Python ints as bitsets.
        cover_mask = [0] * 101
        for idx, mask in enumerate(bad_subsets):
            m = mask
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if 1 <= bit_idx <= 100:
                    cover_mask[bit_idx] |= (1 << idx)
                m ^= lsb
        
        uncovered = (1 << n_bad) - 1
        R = set()
        
        while uncovered:
            best_bit = 0
            max_count = 0
            for i in range(1, 101):
                # Count how many uncovered subsets are covered by i
                count = (uncovered & cover_mask[i]).bit_count()
                if count > max_count:
                    max_count = count
                    best_bit = i
            
            if best_bit == 0:
                # Should not happen if uncovered != 0
                break
                
            R.add(best_bit)
            uncovered &= ~cover_mask[best_bit]
            
        return R

    def local_search(best, deadline):
        """Improve via swap and add moves"""
        best = set(best)
        improved = True
        # Use a random seed for diversity in tie-breaking if needed, 
        # but deterministic is better for reproducibility unless we restart.
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements
            # Sort by value? Or random?
            # Let's try all, break on first success
            for x in range(1, 101):
                if x in best:
                    continue
                candidate = best | {x}
                if is_valid(candidate):
                    best = candidate
                    improved = True
                    break
            
            if time.time() >= deadline:
                break
            
            # Try swapping: remove one, add one
            # To make it faster, we can iterate removals and check if any addition works
            for remove_x in sorted(best):
                if time.time() >= deadline:
                    break
                remaining = best - {remove_x}
                for add_x in range(1, 101):
                    if add_x in remaining:
                        continue
                    candidate = remaining | {add_x}
                    if is_valid(candidate):
                        best = candidate
                        improved = True
                        break
                if improved:
                    break
        return best

    # Strategy 1: Hitting Set based initialization
    best = set()
    
    bad_subsets = generate_bad_subsets()
    if bad_subsets is not None and len(bad_subsets) > 0:
        R = greedy_hitting_set(bad_subsets)
        candidate_S = set(range(1, 101)) - R
        if is_valid(candidate_S):
            best = candidate_S
    
    # If Hitting Set failed or didn't produce a valid set (shouldn't happen if logic is correct),
    # fall back to greedy heuristics
    if not best:
        def greedy(order, existing=None):
            if existing is None:
                existing = set()
            reachable = 1
            selected = set(existing)
            for x in order:
                if x in selected:
                    continue
                new_reachable = reachable | (reachable << x)
                if not ((new_reachable >> 200) & 1):
                    selected.add(x)
                    reachable = new_reachable
            return selected

        candidates = []
        candidates.append(greedy(range(100, 0, -1)))
        candidates.append(greedy(range(1, 101)))
        candidates.append(greedy(sorted(range(1, 101), key=lambda x: -x)))
        
        random.seed(42)
        for i in range(5):
            order = list(range(1, 101))
            random.shuffle(order)
            candidates.append(greedy(order))
            
        best = max(candidates, key=len)
        
    # Local search to improve
    best = local_search(best, DEADLINE - 2.0)
    
    # Try multiple local search restarts if time permits
    if time.time() < DEADLINE - 5.0:
        # We can try restarting from slightly perturbed versions of best
        for _ in range(3):
            if time.time() >= DEADLINE - 2.0:
                break
            # Perturb: remove 2 random elements, then local search
            if len(best) > 2:
                to_remove = random.sample(list(best), 2)
                perturbed = best - set(to_remove)
                improved = local_search(perturbed, DEADLINE - 2.0)
                if len(improved) > len(best):
                    best = improved

    # Final attempt: try to add any remaining elements
    for x in range(1, 101):
        if time.time() >= DEADLINE - 0.5:
            break
        if x in best:
            continue
        candidate = best | {x}
        if is_valid(candidate):
            best = candidate
    
    # Final verification
    assert len(best) > 0
    assert all(1 <= x <= 100 for x in best)
    assert len(best) == len(set(best))

    # Verify no subset sums to 200
    reachable = 1
    for x in sorted(best):
        new_reachable = reachable | (reachable << x)
        if (new_reachable >> 200) & 1:
            return list(best)
        reachable = new_reachable

    return list(best)
