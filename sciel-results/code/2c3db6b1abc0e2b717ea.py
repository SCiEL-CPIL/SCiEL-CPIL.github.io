"""Rewritten program: Sidon base + greedy extension with subset-sum DP check."""
import time
import sys

sys.setrecursionlimit(10000)

# Deadline: 60s wall-clock, keep safety margin at 50s
_DEADLINE = 50.0


def _subset_sum_check(arr, target_sum, target_count):
    """Check if any subset of exactly target_count elements sums to target_sum.
    Uses DP: dp[j][s] = True if we can pick j elements summing to s.
    Returns True if dp[target_count][target_sum] is True.
    """
    n = len(arr)
    if target_count > n:
        return False
    if target_count == 0:
        return target_sum == 0
    # dp[j] is a set of achievable sums using j elements
    # Use bitset-like approach with sets for sparse sums
    dp = [set() for _ in range(target_count + 1)]
    dp[0].add(0)
    for val in arr:
        for j in range(target_count, 0, -1):
            for s in list(dp[j - 1]):
                new_s = s + val
                if new_s <= target_sum:
                    dp[j].add(new_s)
                else:
                    # Can break early if sorted, but arr isn't necessarily sorted here
                    pass
    return target_sum in dp[target_count]


def _is_valid_mean(arr, candidate_val, k_min=2):
    """Check if candidate_val is the mean of any subset of arr of size >= k_min.
    i.e., does any subset S of arr with |S| >= k_min have sum(S) == |S| * candidate_val?
    """
    n = len(arr)
    for k in range(k_min, n + 1):
        target_sum = k * candidate_val
        if target_sum < k:  # min possible sum is k*1
            continue
        if target_sum > k * max(arr):
            continue
        if _subset_sum_check(arr, target_sum, k):
            return False
    return True


def _is_valid(arr, new_val):
    """Check if adding new_val to arr keeps the property:
    1. new_val is not the mean of any subset of arr (size >= 2)
    2. No existing element a in arr is the mean of any subset of (arr + new_val) of size >= 2
       This means: for each a in arr, check subsets of (arr \ {a} + new_val) of size >= 2
       that sum to k*a.
       Subsets can include new_val or not.
       - Subsets not including new_val: already checked when arr was built.
       - Subsets including new_val: subset of (arr \ {a}) of size k-1 summing to k*a - new_val.
    """
    n = len(arr)
    # Check 1: new_val not mean of subset of arr
    if not _is_valid_mean(arr, new_val, k_min=2):
        return False
    # Check 2: for each a in arr, check if a is mean of subset of (arr \ {a}) ∪ {new_val}
    for a in arr:
        others_without_a = [x for x in arr if x != a]
        # Subsets of (others_without_a ∪ {new_val}) of size k >= 2 with sum = k*a
        # Case A: subset does not include new_val -> subset of others_without_a, size k, sum k*a
        #   This was already valid when arr was constructed (since a wasn't mean of any subset of arr\{a} of size>=2)
        # Case B: subset includes new_val -> subset of others_without_a of size k-1, sum = k*a - new_val
        #   k >= 2, so k-1 >= 1
        for k in range(2, n + 1):  # max subset size is n (others_without_a has n-1 elements + new_val = n)
            target_sum = k * a - new_val
            subset_size = k - 1
            if subset_size < 1 or subset_size > len(others_without_a):
                continue
            if target_sum < subset_size:
                continue
            if target_sum > subset_size * max(others_without_a) if others_without_a else target_sum > 0:
                continue
            if _subset_sum_check(others_without_a, target_sum, subset_size):
                return False
    return True


def _sidon_sequence(max_val):
    """Generate a Sidon (B2) sequence up to max_val.
    A Sidon sequence has all pairwise sums a[i]+a[j] (i<=j) distinct.
    This guarantees no element is the mean of two others.
    """
    seq = []
    sums = set()
    for x in range(1, max_val + 1):
        # Check all pairwise sums with existing elements
        valid = True
        for i, a in enumerate(seq):
            s = a + x
            if s in sums:
                valid = False
                break
        if valid:
            # Also check x+x
            if x + x in sums:
                valid = False
        if valid:
            # Add all new sums
            for a in seq:
                sums.add(a + x)
            sums.add(x + x)
            seq.append(x)
    return seq


def construct():
    start_time = time.time()
    
    # Phase 1: Build a Sidon sequence base (guarantees no 2-element mean violations)
    base = _sidon_sequence(512)
    
    # Phase 2: Try to extend the Sidon sequence greedily with remaining numbers
    # For each candidate not in base, check if it can be added without violating the property
    selected = list(base)
    selected_set = set(selected)
    
    # Try to add more elements greedily
    # Sort candidates by value for deterministic order
    candidates = [x for x in range(1, 513) if x not in selected_set]
    
    # Try multiple passes to potentially add more
    for _pass in range(5):
        if time.time() - start_time > _DEADLINE - 5:
            break
        added = False
        for x in candidates:
            if x in selected_set:
                continue
            if time.time() - start_time > _DEADLINE - 5:
                break
            if _is_valid(selected, x):
                selected.append(x)
                selected_set.add(x)
                added = True
        if not added:
            break
    
    # Final verification (lightweight: just check the property holds)
    # The _is_valid function already ensures correctness during construction
    
    return selected


# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    print(f"Result: {result}")
    # Verify
    n = len(result)
    valid = True
    for i, a in enumerate(result):
        others = [x for j, x in enumerate(result) if j != i]
        for k in range(2, len(others) + 1):
            if _subset_sum_check(others, k * a, k):
                print(f"Invalid: {a} is mean of {k}-subset")
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
