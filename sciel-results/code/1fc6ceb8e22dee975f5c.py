# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 82  # safety margin before 90s kill
    n = 512
    max_size = 48

    # Bitset-based subset sum with exact count
    # dp[k] is a bitset where bit s is set if we can pick exactly k elements summing to s
    # We use Python integers as bitsets for speed
    
    def can_pick_exact_count(elements, k, target_sum):
        """Check if we can pick exactly k elements from elements summing to target_sum.
        Returns True/False. Uses bitset DP."""
        if k == 0:
            return target_sum == 0
        if k > len(elements):
            return False
        # Quick bounds check
        if k <= len(elements):
            sorted_elems = sorted(elements)
            min_sum = sum(sorted_elems[:k])
            max_sum = sum(sorted_elems[-k:])
            if target_sum < min_sum or target_sum > max_sum:
                return False
        # Bitset DP
        # dp[j] is an integer bitset; bit s set means sum s achievable with j elements
        dp = [0] * (k + 1)
        dp[0] = 1  # sum 0 achievable with 0 elements
        for val in elements:
            # Update in reverse to avoid using same element twice
            for j in range(min(k - 1, len(elements) - 1), -1, -1):
                # dp[j] << val: shift the bitset for j elements by val
                # Mask to target_sum bits to keep size manageable
                dp[j + 1] |= (dp[j] << val)
                # We only care about bits up to target_sum
                mask = (1 << (target_sum + 1)) - 1
                dp[j + 1] &= mask
        # Check if target_sum bit is set in dp[k]
        return (dp[k] >> target_sum) & 1 == 1

    def is_mean_of_subset(target, others, max_k_check=15):
        """Check if target is the mean of any subset of others with size >= 2 and <= max_k_check."""
        m = len(others)
        if m < 2:
            return False
        upper_k = min(m, max_k_check)
        for k in range(2, upper_k + 1):
            target_sum = k * target
            if can_pick_exact_count(others, k, target_sum):
                return True
        return False

    def is_valid_set(s):
        """Check if set s is valid: no element is the mean of any subset of at least two OTHER elements."""
        s_list = list(s)
        m = len(s_list)
        for i, a in enumerate(s_list):
            others = [s_list[j] for j in range(m) if j != i]
            if is_mean_of_subset(a, others, max_k_check=min(m - 1, 15)):
                return False
        return True

    def full_validate(s):
        """Full validation checking all subset sizes up to len(s)-1."""
        s_list = list(s)
        m = len(s_list)
        for i, a in enumerate(s_list):
            others = [s_list[j] for j in range(m) if j != i]
            # Check all k from 2 to m-1
            for k in range(2, m):
                target_sum = k * a
                if can_pick_exact_count(others, k, target_sum):
                    return False
        return True

    def greedy_construct(ordering, max_k_check=12, deadline=deadline):
        """Greedy construction given an ordering of candidates."""
        selected = []
        selected_set = set()
        for x in ordering:
            if time.time() > deadline:
                break
            if len(selected) >= max_size:
                break
            # Check 1: x is not the mean of any subset of selected (size >= 2)
            if len(selected) >= 2:
                if is_mean_of_subset(x, selected, max_k_check):
                    continue
            # Check 2: for each s in selected, s is not the mean of a subset of (selected - {s}) ∪ {x} of size >= 2
            # A violating subset must include x (by induction, no violation in selected alone)
            # So: x + sum(T) = k*s for some T ⊆ (selected - {s}), |T| = k-1 >= 1
            ok = True
            for s in selected:
                others_without_s = [y for y in selected if y != s]
                m = len(others_without_s)
                # k ranges from 2 to m+1, |T| = k-1 ranges from 1 to m
                for t_size in range(1, min(m, max_k_check) + 1):
                    k = t_size + 1
                    target_t_sum = k * s - x
                    if target_t_sum < 0:
                        continue
                    # Quick bounds
                    if t_size <= m:
                        sorted_o = sorted(others_without_s)
                        min_t = sum(sorted_o[:t_size])
                        max_t = sum(sorted_o[-t_size:])
                        if target_t_sum < min_t or target_t_sum > max_t:
                            continue
                    if can_pick_exact_count(others_without_s, t_size, target_t_sum):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.append(x)
                selected_set.add(x)
        return selected

    def local_search_improve(s, deadline=deadline):
        """Try to improve set s by swapping elements."""
        s = list(s)
        s_set = set(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing one element and adding another
            for i in range(len(s)):
                if time.time() > deadline:
                    break
                candidate_remove = s[i]
                s_without = [x for x in s if x != candidate_remove]
                s_without_set = s_set - {candidate_remove}
                # Try to add a new element
                for x in range(1, n + 1):
                    if time.time() > deadline:
                        break
                    if x in s_without_set:
                        continue
                    if len(s_without) >= 2 and is_mean_of_subset(x, s_without, max_k_check=10):
                        continue
                    # Check if any s in s_without is now a mean involving x
                    ok = True
                    for sv in s_without:
                        others_without_sv = [y for y in s_without if y != sv]
                        m = len(others_without_sv)
                        for t_size in range(1, min(m, 10) + 1):
                            k = t_size + 1
                            target_t_sum = k * sv - x
                            if target_t_sum < 0:
                                continue
                            if t_size <= m:
                                sorted_o = sorted(others_without_sv)
                                min_t = sum(sorted_o[:t_size])
                                max_t = sum(sorted_o[-t_size:])
                                if target_t_sum < min_t or target_t_sum > max_t:
                                    continue
                            if can_pick_exact_count(others_without_sv, t_size, target_t_sum):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        s = s_without + [x]
                        s.sort()
                        s_set = set(s)
                        improved = True
                        break
                if improved:
                    break
        return s

    # Multi-start greedy with different orderings
    best_set = []
    best_size = 0

    # Ordering 1: natural order
    ordering1 = list(range(1, n + 1))
    s1 = greedy_construct(ordering1, max_k_check=10, deadline=deadline)
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)

    # Ordering 2: reverse
    if time.time() < deadline - 10:
        ordering2 = list(range(n, 0, -1))
        s2 = greedy_construct(ordering2, max_k_check=10, deadline=deadline)
        if len(s2) > best_size:
            best_set = s2
            best_size = len(s2)

    # Ordering 3: random
    if time.time() < deadline - 10:
        random.seed(42)
        ordering3 = list(range(1, n + 1))
        random.shuffle(ordering3)
        s3 = greedy_construct(ordering3, max_k_check=10, deadline=deadline)
        if len(s3) > best_size:
            best_set = s3
            best_size = len(s3)

    # Ordering 4: prioritize numbers with fewer "conflicts"
    if time.time() < deadline - 15:
        # Count for each x how many pairs (a,b) in 1..512 have a+b=2x
        # More conflicts = harder to include
        conflict_count = [0] * (n + 1)
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                s = a + b
                if s % 2 == 0:
                    mid = s // 2
                    if 1 <= mid <= n:
                        conflict_count[mid] += 1
        ordering4 = sorted(range(1, n + 1), key=lambda x: conflict_count[x])
        s4 = greedy_construct(ordering4, max_k_check=10, deadline=deadline)
        if len(s4) > best_size:
            best_set = s4
            best_size = len(s4)

    # Try more random orderings
    if time.time() < deadline - 10:
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            random.seed(seed * 137 + 7)
            ordering = list(range(1, n + 1))
            random.shuffle(ordering)
            s = greedy_construct(ordering, max_k_check=10, deadline=deadline)
            if len(s) > best_size:
                best_set = s
                best_size = len(s)

    # Local search improvement
    if best_size < max_size and time.time() < deadline - 5:
        improved = local_search_improve(best_set, deadline=deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)

    # Final validation with full check
    if best_set and time.time() < deadline - 2:
        if not full_validate(best_set):
            # Remove elements until valid
            best_set = list(best_set)
            changed = True
            while changed and best_set:
                changed = False
                for i in range(len(best_set)):
                    if time.time() > deadline - 1:
                        break
                    trial = best_set[:i] + best_set[i+1:]
                    if full_validate(trial):
                        best_set = trial
                        changed = True
                        break

    return best_set
# EVOLVE-BLOCK-END
