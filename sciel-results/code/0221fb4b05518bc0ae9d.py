import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = 512
    max_size = 48

    # Precompute conflict data using bitset DP
    # For each candidate x in 1..512, we need to know which subsets of other elements
    # have mean exactly x.
    
    # Strategy: For a set S, element a is invalid if there exists a subset T of S\{a},
    # |T| >= 2, such that sum(T) = |T| * a.
    # Equivalently, for each a, we check if any subset of S\{a} has sum = k*a for some k>=2.
    
    # Precompute for each target a: the set of "conflict" elements
    # Element b conflicts with a (as mean) if b can be part of a subset whose mean is a.
    # But this is complex. Instead, let's use a different approach.
    
    # For greedy construction with fast validation:
    # Maintain a set S. When adding x, check:
    # 1. x is not the mean of any subset of S of size >= 2
    # 2. No element of S is the mean of any subset of (S ∪ {x}) \ {that element} of size >= 2
    
    # For check 1: need to verify no subset of S has sum = k*x for k >= 2
    # For check 2: for each a in S, need to verify no subset of (S ∪ {x}) \ {a} has sum = k*a
    
    # Use bitset DP for subset sum checks
    def subset_sum_bitset(elements, target_sum, min_count=2):
        """Check if any subset of elements with size >= min_count sums to target_sum.
        Returns True if such subset exists."""
        # dp[s] = bitmask of achievable subset sizes for sum s
        # But target_sum can be up to 48*512 = 24576, too large for full DP
        # Instead, use a different approach: track achievable (sum, count) pairs
        # Actually, let's use: for each possible count k, track achievable sums
        # But that's expensive.
        
        # Alternative: since we only need to check specific targets, use meet-in-the-middle
        # or a simpler DP.
        
        # For efficiency, use DP with bitsets for counts
        # dp[c] is a set of achievable sums with exactly c elements
        # But this can be expensive.
        
        # Let's use a different approach: for the specific target, do a bounded DP
        # We want to know if there's a subset of size k >= 2 with sum = target
        # This is equivalent to: is target achievable as sum of k distinct elements for some k >= 2?
        
        # Use DP: reachable[s] = set of counts achievable for sum s
        # But s goes up to target, and we only care about s = target
        
        # Better: use a 2D approach but only track up to target
        # dp[s] = bitmask where bit c is set if sum s is achievable with c elements
        max_s = target_sum
        if max_s > 25000:  # too large
            return False  # fallback: assume valid (shouldn't happen for our targets)
        
        dp = [0] * (max_s + 1)
        dp[0] = 1  # 0 elements, sum 0
        for elem in elements:
            if elem > max_s:
                continue
            # Update in reverse to avoid using element twice
            for s in range(max_s, elem - 1, -1):
                if dp[s - elem]:
                    # Shift left by 1 to increment count
                    dp[s] |= (dp[s - elem] << 1)
        
        # Check if any count >= 2 is achievable for sum = target_sum
        if target_sum <= max_s:
            counts = dp[target_sum]
            # Check if any bit >= 2 is set
            if counts >> 2:  # any bit from position 2 onwards
                return True
        return False

    def is_valid_addition(S, x):
        """Check if adding x to set S keeps it valid."""
        if x in S:
            return False
        
        S_list = sorted(S)
        
        # Check 1: x is not the mean of any subset of S of size >= 2
        # i.e., no subset of S has sum = k*x for k >= 2
        for k in range(2, len(S_list) + 1):
            target = k * x
            if target > sum(S_list):
                break
            if subset_sum_bitset(S_list, target, 2):
                return False
        
        # Check 2: for each a in S, a is not the mean of any subset of (S ∪ {x}) \ {a} of size >= 2
        for a in S_list:
            others = [b for b in S_list if b != a] + ([x] if x != a else [])
            for k in range(2, len(others) + 1):
                target = k * a
                if target > sum(others):
                    break
                if subset_sum_bitset(others, target, 2):
                    return False
        
        return True

    # Precompute conflict counts for ordering
    # For each element v, count how many other elements u could form a conflict with v
    # as a mean. This is heuristic: if v is close to many pairs, it's more constrained.
    
    def compute_conflict_score(v, all_elements):
        """Heuristic score: how many potential conflicts v has."""
        score = 0
        for u in all_elements:
            if u == v:
                continue
            # v could be mean of {u, w} for some w: w = 2v - u
            w = 2 * v - u
            if w in all_elements and w != u and w != v:
                score += 1
            # v could be mean of {u, w, z}: harder to count
        return score

    # Greedy with conflict-aware ordering
    best_set = set()
    
    # Initial seed: powers of 2
    powers_of_2 = [2**i for i in range(0, 10) if 2**i <= n]
    for p in powers_of_2:
        if len(best_set) < max_size and is_valid_addition(best_set, p):
            best_set.add(p)

    # Compute conflict scores for all candidates
    all_candidates = list(range(1, n + 1))
    conflict_scores = {v: compute_conflict_score(v, set(all_candidates)) for v in all_candidates}
    
    # Try multiple random orderings weighted by inverse conflict score
    random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        if len(best_set) >= max_size:
            break
        
        current_set = set(best_set)
        
        # Create ordering: elements with fewer conflicts first
        candidates = [v for v in all_candidates if v not in current_set]
        
        if attempt == 0:
            # Order by conflict score (fewer conflicts first)
            candidates.sort(key=lambda v: conflict_scores[v])
        else:
            # Random shuffle with bias
            candidates.sort(key=lambda v: (conflict_scores[v], random.random()))
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current_set) >= max_size:
                break
            if is_valid_addition(current_set, x):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = set(current_set)
    
    # Local search: try swapping elements
    if time.time() < deadline - 5:
        current_set = set(best_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for a in list(current_set):
                if time.time() > deadline - 2:
                    break
                current_set.remove(a)
                # Try to add a new element
                for x in range(1, n + 1):
                    if x not in current_set and is_valid_addition(current_set, x):
                        current_set.add(x)
                        improved = True
                        break
                if len(current_set) >= len(best_set):
                    best_set = set(current_set)
                current_set.add(a)  # restore
                if improved:
                    break
    
    return sorted(list(best_set))
