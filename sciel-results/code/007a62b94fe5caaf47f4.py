"""Rewritten program: bitset-based subset-sum + conflict-scored greedy."""
import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = 512
    MAX_K = 48  # max subset size to check (we only have up to 48 elements)

    def subset_sum_bitset(values, target, max_count):
        """
        Returns True if there exists a subset of `values` of size exactly `max_count`
        (or at most, but we want exactly) summing to `target`.
        Actually, we need: exists subset of size k (for some k in [2, max_count]) with sum = k*target_mean.
        But here we call with a specific k and target_sum.
        """
        # dp[c] is a bitset where bit s is set if sum s is achievable with c elements
        # We want to know if target is achievable with exactly max_count elements
        # Use: dp[c] as int, bit i set means sum i achievable with c elements
        # But sums can be large. Max sum = max_count * max(values)
        max_sum = max_count * max(values) if values else 0
        if target > max_sum or target < 0:
            return False
        
        # dp[c] = bitset of achievable sums with c elements
        dp = [0] * (max_count + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for val in values:
            # Update in reverse to avoid using same element twice
            for c in range(max_count - 1, -1, -1):
                # dp[c] shifted left by val, masked to max_sum
                dp[c + 1] |= (dp[c] << val)
        
        # Check if target bit is set in dp[max_count]
        return (dp[max_count] >> target) & 1 == 1

    def is_mean_of_subset(target, others):
        """
        Check if `target` is the mean of any subset of `others` with size >= 2.
        i.e., exists k >= 2 and subset S of others with |S| = k and sum(S) = k * target.
        """
        m = len(others)
        if m < 2:
            return False
        
        # Sort others for bounds
        others_sorted = sorted(others)
        
        for k in range(2, min(m, MAX_K) + 1):
            target_sum = k * target
            # Quick bounds
            min_sum = sum(others_sorted[:k])
            max_sum = sum(others_sorted[-k:])
            if target_sum < min_sum or target_sum > max_sum:
                continue
            
            if subset_sum_bitset(others, target_sum, k):
                return True
        
        return False

    def check_violation_new(new_elem, selected):
        """
        Check if adding new_elem to selected creates any violation.
        Two types:
        1. new_elem is mean of subset of selected (size >= 2)
        2. Some s in selected is mean of subset of (selected - {s}) ∪ {new_elem} (size >= 2)
           This reduces to: exists k >= 2, subset T of (selected - {s}) with |T| = k-1 >= 1
           and new_elem + sum(T) = k * s, i.e., sum(T) = k*s - new_elem.
        """
        # Check 1: new_elem is mean of subset of selected
        if len(selected) >= 2:
            if is_mean_of_subset(new_elem, selected):
                return True
        
        # Check 2: for each s in selected, s is mean of subset involving new_elem
        for s in selected:
            others = [y for y in selected if y != s]
            m = len(others)
            if m < 1:
                continue
            
            # k ranges from 2 to m+1 (subset size including new_elem)
            # |T| = k-1, so |T| ranges from 1 to m
            for k in range(2, m + 2):
                t_size = k - 1
                if t_size < 1 or t_size > m:
                    continue
                target_t_sum = k * s - new_elem
                if target_t_sum < 0:
                    continue
                
                # Bounds
                others_sorted = sorted(others)
                min_t_sum = sum(others_sorted[:t_size])
                max_t_sum = sum(others_sorted[-t_size:])
                if target_t_sum < min_t_sum or target_t_sum > max_t_sum:
                    continue
                
                if subset_sum_bitset(others, target_t_sum, t_size):
                    return True
        
        return False

    def conflict_count(x, selected):
        """
        Count how many violations would be created if x were added to selected.
        This is used for greedy ordering: prefer x with fewer conflicts.
        """
        if check_violation_new(x, selected):
            # We could count more precisely, but for now just return a large number
            # Actually, let's count the number of (s, k) pairs that would violate
            count = 0
            # Check 1
            if len(selected) >= 2:
                others = selected
                others_sorted = sorted(others)
                for k in range(2, min(len(others), MAX_K) + 1):
                    target_sum = k * x
                    min_sum = sum(others_sorted[:k])
                    max_sum = sum(others_sorted[-k:])
                    if min_sum <= target_sum <= max_sum:
                        if subset_sum_bitset(others, target_sum, k):
                            count += 1
            # Check 2
            for s in selected:
                others = [y for y in selected if y != s]
                m = len(others)
                if m < 1:
                    continue
                others_sorted = sorted(others)
                for k in range(2, m + 2):
                    t_size = k - 1
                    if t_size < 1 or t_size > m:
                        continue
                    target_t_sum = k * s - x
                    if target_t_sum < 0:
                        continue
                    min_t_sum = sum(others_sorted[:t_size])
                    max_t_sum = sum(others_sorted[-t_size:])
                    if min_t_sum <= target_t_sum <= max_t_sum:
                        if subset_sum_bitset(others, target_t_sum, t_size):
                            count += 1
            return count
        return 0

    # Greedy construction with conflict-scored ordering
    selected = []
    candidates = list(range(1, n + 1))
    
    # We'll use a priority approach: at each step, evaluate all remaining candidates
    # and pick the one with the fewest conflicts (and is valid)
    # But this is O(n^2 * check_cost) which might be slow.
    # Alternative: try a few orderings and pick the best.
    
    best_selected = []
    
    # Try natural order first
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if not check_violation_new(x, selected):
            selected.append(x)
    
    if len(selected) > len(best_selected):
        best_selected = selected[:]
    
    # Try reverse order
    selected = []
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if not check_violation_new(x, selected):
            selected.append(x)
    
    if len(selected) > len(best_selected):
        best_selected = selected[:]
    
    # Try odd numbers first, then even
    selected = []
    for x in range(1, n + 1, 2):
        if time.time() > deadline:
            break
        if not check_violation_new(x, selected):
            selected.append(x)
    for x in range(2, n + 1, 2):
        if time.time() > deadline:
            break
        if not check_violation_new(x, selected):
            selected.append(x)
    
    if len(selected) > len(best_selected):
        best_selected = selected[:]
    
    # Try a conflict-scored greedy: at each step, pick the candidate with min conflicts
    # This is expensive but might find better sets
    if time.time() < deadline - 30:
        selected = []
        remaining = set(range(1, n + 1))
        while remaining:
            if time.time() > deadline - 5:
                break
            best_x = None
            best_conf = None
            # Sample a subset of candidates to keep this fast
            sample = list(remaining)
            random.shuffle(sample)
            sample = sample[:min(50, len(sample))]
            for x in sample:
                conf = conflict_count(x, selected)
                if best_conf is None or conf < best_conf:
                    best_conf = conf
                    best_x = x
                if conf == 0:
                    break  # Found a conflict-free choice
            if best_x is not None and best_conf is not None and best_conf == 0:
                selected.append(best_x)
                remaining.discard(best_x)
            elif best_x is not None and best_conf is not None and best_conf < 3:
                # Allow low-conflict additions
                selected.append(best_x)
                remaining.discard(best_x)
            else:
                break
        
        if len(selected) > len(best_selected):
            best_selected = selected[:]
    
    return best_selected
