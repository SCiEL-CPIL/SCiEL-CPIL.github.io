# EVOLVE-BLOCK-START
import time
import random

def _can_add_bitset(S, cand):
    """
    Check if cand can be added to S without violating the mean condition.
    Uses bitset DP for subset sum with cardinality constraints.
    S is a list of existing elements (not including cand).
    Returns True if cand can be safely added.
    """
    n = len(S)
    if n < 1:
        return True
    
    # We need to check two things:
    # 1. cand is not the mean of any subset of S with size >= 2
    #    i.e., no subset T of S with |T| >= 2 and sum(T) = |T| * cand
    # 2. No element a in S is the mean of any subset of (S ∪ {cand}) with size >= 2
    #    that includes cand.
    #    i.e., no subset T of S with |T| >= 1 such that (sum(T) + cand) / (|T| + 1) = a
    #    => sum(T) = a * (|T| + 1) - cand = a * |T| + a - cand
    
    # DP: dp[k] is a bitset where bit s is set if there's a subset of S of size k with sum s.
    # Max sum: n * 512. For n up to 48, max sum ~ 24576.
    # We'll use Python integers as bitsets.
    
    max_sum = n * 512
    # dp[k] is an integer where bit i represents sum i
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements
    
    for x in S:
        # Update in reverse to avoid using x multiple times
        for k in range(n, 0, -1):
            # dp[k] |= dp[k-1] << x
            dp[k] |= (dp[k - 1] << x)
    
    # Check 1: Is cand the mean of any subset of S with size >= 2?
    # Need: exists k >= 2 such that bit (k * cand) is set in dp[k]
    for k in range(2, n + 1):
        target = k * cand
        if target <= max_sum and (dp[k] >> target) & 1:
            return False
    
    # Check 2: For each a in S, is a the mean of a subset including cand?
    # Need: exists a in S, exists k >= 1 such that bit (a*k + a - cand) is set in dp[k]
    # Note: a*k + a - cand must be >= 0 and <= max_sum
    for a in S:
        for k in range(1, n + 1):
            target = a * k + a - cand
            if 0 <= target <= max_sum and (dp[k] >> target) & 1:
                return False
    
    return True

def _conflict_count(x, all_numbers):
    """
    Count how many 2-element arithmetic progressions x could be part of.
    x is the mean: a, x, b where a + b = 2x, a != x, b != x, a != b.
    """
    count = 0
    for a in all_numbers:
        if a == x:
            continue
        b = 2 * x - a
        if b != x and b != a and b in all_numbers:
            count += 1
    return count

def _conflict_score(x, all_numbers):
    """
    Refined conflict score: count both 2-element and 3-element subset conflicts.
    For 2-element: a + b = 2x (x is mean of {a,b})
    For 3-element: a + b + c = 3x (x is mean of {a,b,c}) - approximate by checking
    if there are pairs (a,b) with a+b = 3x - c for some c, but this is expensive.
    Instead, use a weighted combination: 2-element conflicts weighted more heavily
    since they're more common and easier to form.
    """
    count_2 = 0
    for a in all_numbers:
        if a == x:
            continue
        b = 2 * x - a
        if b != x and b != a and b in all_numbers:
            count_2 += 1
    
    # Approximate 3-element conflicts: for each pair (a,b) in all_numbers,
    # check if there exists c such that a+b+c = 3x
    # This is O(n^2) which is too slow for 512 elements, so we use a heuristic:
    # count pairs (a,b) where a+b is close to 3x (within some range)
    # Actually, let's just use 2-element count as primary and add a small penalty
    # for numbers that are "central" (more likely to be means of larger subsets)
    centrality = abs(x - 256)  # distance from center of [1,512]
    # Lower centrality (closer to center) means more potential conflicts
    score = count_2 * 10 + (512 - centrality) // 10
    
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    best_set = []
    best_len = 0
    
    # Precompute conflict scores for all numbers 1..512
    all_numbers = set(range(1, 513))
    conflict_scores = {}
    for x in range(1, 513):
        conflict_scores[x] = _conflict_score(x, all_numbers)
    
    # Sort candidates by conflict score (fewer conflicts first)
    candidates_sorted = sorted(range(1, 513), key=lambda x: conflict_scores[x])
    
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # For the first few restarts, use sorted orders with different tie-breaking
        # For subsequent restarts, use random shuffles with bias toward low-conflict numbers
        if restart == 0:
            candidates = list(candidates_sorted)
        elif restart == 1:
            # Reverse order to explore different structures
            candidates = list(reversed(candidates_sorted))
        elif restart == 2:
            # Sort by conflict score but break ties randomly
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            candidates.sort(key=lambda x: conflict_scores[x])
        else:
            # Random shuffle with bias: low-conflict numbers more likely to come first
            candidates = list(range(1, 513))
            # Assign weights inversely proportional to conflict score
            weights = [1.0 / (1.0 + conflict_scores[x] * 0.01) for x in candidates]
            total_weight = sum(weights)
            # Sample without replacement using weighted random selection
            selected = []
            remaining = list(range(512))
            remaining_weights = weights[:]
            for _ in range(512):
                if not remaining:
                    break
                # Weighted random selection
                r = random.random() * sum(remaining_weights)
                cum = 0
                for i, w in enumerate(remaining_weights):
                    cum += w
                    if r <= cum:
                        selected.append(remaining[i])
                        remaining.pop(i)
                        remaining_weights.pop(i)
                        break
            candidates = selected
        
        current_set = []
        current_set_set = set()
        
        for cand in candidates:
            if time.time() > deadline - 5.0:
                break
            
            if cand in current_set_set:
                continue
            
            if _can_add_bitset(current_set, cand):
                current_set.append(cand)
                current_set_set.add(cand)
        
        if len(current_set) > best_len:
            best_set = current_set
            best_len = len(current_set)
            
            if best_len >= 48:
                return best_set
    
    # Try to add any remaining numbers to best_set
    if best_set:
        best_set_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        # Sort remaining by conflict score
        remaining.sort(key=lambda x: conflict_scores[x])
        
        for cand in remaining:
            if time.time() > deadline - 2.0:
                break
            if _can_add_bitset(best_set, cand):
                best_set.append(cand)
    
    return best_set

# EVOLVE-BLOCK-END
