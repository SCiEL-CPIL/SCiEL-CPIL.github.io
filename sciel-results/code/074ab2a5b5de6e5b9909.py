"""Improved program with conflict graph and greedy coloring for subset mean avoidance."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0
    n = 512
    target_size = 48

    # Precompute 3-term AP conflicts: (a,b) conflict if (a+b) even and midpoint in [1,n]
    # This is a NECESSARY condition for invalidity (subset of size 2)
    # Build as a set of frozenset pairs for fast lookup
    conflict_pairs = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            s = a + b
            if s % 2 == 0:
                mid = s // 2
                if 1 <= mid <= n and mid != a and mid != b:
                    conflict_pairs.add((a, b))
    
    # Also precompute: for each pair (a,b), is there a c in [1,n] \ {a,b} such that c = (a+b)/2?
    # That's the same as above. But we also need: for a given set S, check if any element is mean of 2+ others.
    # Fast necessary check: no 3-term AP within S.
    
    def _has_3ap(selected_set):
        """Check if selected_set contains any 3-term arithmetic progression."""
        s = set(selected_set)
        for a in selected_set:
            for b in selected_set:
                if a == b:
                    continue
                mid = (a + b) // 2
                if (a + b) % 2 == 0 and mid in s and mid != a and mid != b:
                    return True
        return False

    def _build_dp(arr):
        """Build DP table: dp[k] is bitset of all subset sums of size k."""
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val)
        return dp

    def _is_valid_set(selected):
        """Full validation: no element is mean of any subset of >=2 other elements."""
        m = len(selected)
        if m < 3:
            return True
        for i, a in enumerate(selected):
            others = [selected[j] for j in range(m) if j != i]
            dp_others = _build_dp(others)
            total_others = sum(others)
            for k in range(2, len(others) + 1):
                target = k * a
                if target > total_others:
                    break
                if (dp_others[k] >> target) & 1:
                    return False
        return True

    def _is_valid_incremental_fast(existing, x, existing_dp=None, existing_sum=None):
        """
        Check if adding x to existing keeps validity.
        If existing_dp and existing_sum provided, use them; else compute.
        """
        if not existing:
            return True
        m = len(existing)
        if existing_dp is None:
            existing_dp = _build_dp(existing)
        if existing_sum is None:
            existing_sum = sum(existing)
        
        # Check 1: x is not mean of any subset of existing of size >= 2
        for k in range(2, m + 1):
            target = k * x
            if target > existing_sum:
                break
            if (existing_dp[k] >> target) & 1:
                return False
        
        # Check 2: no element a in existing is mean of subset including x
        # (sum of t from existing + x) / (t+1) = a => sum = (t+1)*a - x
        for a in existing:
            for t in range(1, m + 1):
                target = (t + 1) * a - x
                if target < 0:
                    continue
                if target > existing_sum:
                    break
                if (existing_dp[t] >> target) & 1:
                    return False
        return True

    def _remove_from_dp(dp, val, m):
        """Remove val from DP table. Returns new dp for m-1 elements."""
        # This is tricky with bitsets. Instead, rebuild.
        # For efficiency, we'll just rebuild when needed.
        pass

    # Strategy: Start with a large random set with no 3-AP, then hill-climb
    # to find the largest valid set.
    
    best = []
    best_size = 0

    def _try_add_to_selected(selected, candidate, sel_dp, sel_sum):
        """Try to add candidate to selected. Returns (new_selected, new_dp, new_sum) or None."""
        if candidate in selected:
            return None
        if not _is_valid_incremental_fast(selected, candidate, sel_dp, sel_sum):
            return None
        new_selected = selected + [candidate]
        new_sum = sel_sum + candidate
        new_dp = _build_dp(new_selected)
        return (new_selected, new_dp, new_sum)

    def _greedy_build(order, max_size=target_size):
        """Greedy build from given order."""
        selected = []
        sel_dp = [1]
        sel_sum = 0
        for x in order:
            if len(selected) >= max_size:
                break
            result = _try_add_to_selected(selected, x, sel_dp, sel_sum)
            if result:
                selected, sel_dp, sel_sum = result
        return selected

    # Phase 1: Multiple greedy builds with different orderings
    # Ordering 1: Numbers with fewest 3-AP conflicts
    conflict_count = [0] * (n + 1)
    for a, b in conflict_pairs:
        conflict_count[a] += 1
        conflict_count[b] += 1
    
    order_by_conflict = sorted(range(1, n + 1), key=lambda x: (conflict_count[x], x))
    result = _greedy_build(order_by_conflict)
    if len(result) > best_size:
        best = result
        best_size = len(result)
        if best_size >= target_size:
            return best

    # Ordering 2: Natural
    result = _greedy_build(list(range(1, n + 1)))
    if len(result) > best_size:
        best = result
        best_size = len(result)
        if best_size >= target_size:
            return best

    # Ordering 3: Descending
    result = _greedy_build(list(range(n, 0, -1)))
    if len(result) > best_size:
        best = result
        best_size = len(result)
        if best_size >= target_size:
            return best

    # Ordering 4: Odd numbers first (fewer 3-AP since (odd+odd)/2 = odd)
    odds = list(range(1, n + 1, 2))
    evens = list(range(2, n + 1, 2))
    result = _greedy_build(odds + evens)
    if len(result) > best_size:
        best = result
        best_size = len(result)
        if best_size >= target_size:
            return best

    # Ordering 5: Primes first
    def _is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    primes = [x for x in range(1, n + 1) if _is_prime(x)]
    non_primes = [x for x in range(1, n + 1) if not _is_prime(x)]
    result = _greedy_build(primes + non_primes)
    if len(result) > best_size:
        best = result
        best_size = len(result)
        if best_size >= target_size:
            return best

    # Phase 2: Hill-climbing from best set
    # Try to improve by: (a) adding a new element, (b) swapping one element for another
    def _hill_climb(start_set, max_rounds=50):
        selected = list(start_set)
        sel_dp = _build_dp(selected)
        sel_sum = sum(selected)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            if time.time() > deadline - 5.0:
                break
            improved = False
            rounds += 1
            
            # Try adding any element
            for x in range(1, n + 1):
                if x in selected:
                    continue
                if _is_valid_incremental_fast(selected, x, sel_dp, sel_sum):
                    selected.append(x)
                    sel_sum += x
                    sel_dp = _build_dp(selected)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            for i in range(len(selected)):
                if time.time() > deadline - 5.0:
                    break
                to_remove = selected[i]
                remaining = selected[:i] + selected[i+1:]
                rem_sum = sel_sum - to_remove
                rem_dp = _build_dp(remaining)
                for x in range(1, n + 1):
                    if x in remaining or x == to_remove:
                        continue
                    if _is_valid_incremental_fast(remaining, x, rem_dp, rem_sum):
                        new_selected = remaining + [x]
                        # Verify full validity
                        if _is_valid_set(new_selected):
                            selected = new_selected
                            sel_dp = _build_dp(selected)
                            sel_sum = sum(selected)
                            improved = True
                            break
                if improved:
                    break
        
        return selected

    if best_size < target_size:
        improved = _hill_climb(best, max_rounds=30)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
            if best_size >= target_size:
                return best

    # Phase 3: Random restarts with hill-climbing
    for trial in range(60):
        if time.time() > deadline - 10.0:
            break
        random.seed(42 + trial)
        
        # Build a random initial set with no 3-AP
        random_set = set()
        for x in random.sample(range(1, n + 1), min(25, n)):
            test_set = random_set | {x}
            if not _has_3ap(list(test_set)):
                random_set.add(x)
            if len(random_set) >= 25:
                break
        
        # Greedy extend
        remaining = [x for x in range(1, n + 1) if x not in random_set]
        random.shuffle(remaining)
        order = list(random_set) + remaining
        result = _greedy_build(order)
        
        if len(result) > best_size:
            best = result
            best_size = len(result)
            if best_size >= target_size:
                return best
        
        # Hill-climb from this result
        if len(result) >= best_size - 2 and time.time() < deadline - 5.0:
            improved = _hill_climb(result, max_rounds=15)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
                if best_size >= target_size:
                    return best

    # Phase 4: Final attempt - try to fill gaps
    if best_size < target_size and time.time() < deadline - 3.0:
        # Sort best and try to add missing numbers
        best_set = set(best)
        sel_dp = _build_dp(best)
        sel_sum = sum(best)
        for x in range(1, n + 1):
            if x in best_set:
                continue
            if _is_valid_incremental_fast(best, x, sel_dp, sel_sum):
                best.append(x)
                best_set.add(x)
                sel_sum += x
                sel_dp = _build_dp(best)
                if len(best) >= target_size:
                    break

    return best
# EVOLVE-BLOCK-END
