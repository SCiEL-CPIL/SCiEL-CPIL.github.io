"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations

def construct():
    # We want to construct a set of distinct integers in 1..256 such that:
    # 1. All nonempty subset sums are distinct (no two subsets have the same sum).
    #    - This is equivalent to the set being "sum-distinct" or having unique subset sums.
    #    - A sufficient condition: each element is greater than the sum of all previous elements.
    #      This ensures that each subset sum is unique (binary representation analogy).
    # 2. No two distinct subset sums divide one another.
    #    - If s1 and s2 are subset sums with s1 < s2, then s1 must not divide s2.
    #
    # Strategy:
    # - Use a greedy construction where each new element is larger than the sum of all previous.
    #   This guarantees unique subset sums.
    # - Additionally, we need to ensure that no subset sum divides another.
    #   Since the smallest subset sum is the smallest element, and the largest is the total sum,
    #   we need to check all pairs of subset sums for divisibility.
    #
    # To maximize size, we want as many elements as possible within 1..256.
    # The minimal sequence with unique subset sums is: 1, 2, 4, 8, 16, ... (powers of 2).
    # But powers of 2 have the problem that 1 divides everything, 2 divides 4, 8, etc.
    # So we need to choose elements carefully to avoid divisibility among subset sums.
    #
    # Let's try a different approach: start with a base set and greedily add elements
    # that maintain both properties.
    #
    # Actually, let's think more carefully. If we use a superincreasing sequence (each element
    # > sum of all previous), then subset sums are unique. The subset sums range from the
    # smallest element to the total sum. We need no two subset sums to divide each other.
    #
    # Let's try: start with a small element, then add elements that are large enough to be
    # superincreasing, and also avoid creating divisibility relations among subset sums.
    #
    # For simplicity and to ensure correctness within time limits, let's use a precomputed
    # set that satisfies the conditions. We can verify it programmatically.
    #
    # Let's try to build a set greedily:
    # - Start with empty set.
    # - For each candidate from 1 to 256, check if adding it maintains:
    #   a) Unique subset sums (if we maintain superincreasing, this is automatic)
    #   b) No two subset sums divide each other.
    #
    # But checking all subset sums for each candidate is expensive. Let's limit the search.
    #
    # Alternative: Use a known construction or heuristic.
    # Let's try elements that are all odd and large, to avoid small divisors.
    #
    # Actually, let's just implement a greedy algorithm with pruning:
    # - Maintain a list of selected elements.
    # - Maintain all current subset sums.
    # - For each candidate in 1..256 (in some order), check if adding it:
    #   * Keeps subset sums unique (new subset sums = old sums + candidate, and none of these
    #     should collide with old sums or each other).
    #   * No new subset sum divides or is divided by any existing subset sum.
    #
    # To keep it fast, we'll try candidates in descending order (larger elements are "safer"
    # for uniqueness but may create divisibility with small sums).
    #
    # Let's implement this.

    def all_subset_sums(elements):
        sums = {0}
        for e in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + e)
            sums = sums | new_sums
        sums.discard(0)
        return sums

    def check_divisibility(sums):
        # Check that no two distinct sums divide each other.
        sum_list = sorted(sums)
        for i in range(len(sum_list)):
            for j in range(i+1, len(sum_list)):
                if sum_list[j] % sum_list[i] == 0:
                    return False
        return True

    # Greedy construction: try to add as many elements as possible.
    # We'll try candidates in a specific order. Let's try descending from 256 to 1.
    selected = []
    current_sums = set()
    
    for candidate in range(256, 0, -1):
        # Try adding candidate
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + candidate)
        new_sums.add(candidate)
        
        # Check uniqueness: new_sums should have size = len(current_sums) * 2 + 1 - 1 (since 0 is removed)
        # Actually, if current_sums has k elements (excluding 0), then new_sums should have 2k+1 elements
        # (the old k, the new k+1 including candidate, but wait...)
        # Let's just check: len(new_sums) should be len(current_sums) + len(current_sums) + 1
        # No, that's not right. Let me recount.
        # If we have sums S (nonempty), and we add element e:
        # New sums = S ∪ {s + e for s in S} ∪ {e}
        # If all are distinct, then |new_sums| = |S| + |S| + 1 = 2|S| + 1
        # But wait, that's only if e is not in S and no s+e is in S or equal to another s'+e.
        # With superincreasing, this is guaranteed.
        
        if len(new_sums) != 2 * len(current_sums) + 1:
            continue
        
        # Check divisibility
        if not check_divisibility(new_sums):
            continue
        
        selected.append(candidate)
        current_sums = new_sums
    
    # The greedy above may not find the maximum. Let's try a different order.
    # Try ascending order but with a twist: prefer elements that are "safe".
    
    # Actually, let's try a different strategy: use a fixed set that we know works.
    # Let me manually construct a set.
    
    # Start with a large element to avoid small divisors.
    # Try: 128, 64, 32, 16, 8, 4, 2, 1 -> but this has divisibility issues.
    
    # Let's try: 100, 101, 102, ... but these may not be superincreasing.
    
    # I'll implement a more careful greedy with multiple restarts and different orders.
    
    best = []
    best_sums = set()
    
    orders = [
        list(range(256, 0, -1)),
        list(range(1, 257)),
        [256, 128, 64, 32, 16, 8, 4, 2, 1],  # powers of 2 descending
        [1, 2, 4, 8, 16, 32, 64, 128, 256],  # powers of 2 ascending
    ]
    
    for order in orders:
        selected = []
        current_sums = set()
        for candidate in order:
            if candidate in selected:
                continue
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + candidate)
            new_sums.add(candidate)
            
            if len(new_sums) != 2 * len(current_sums) + 1:
                continue
            
            if not check_divisibility(new_sums):
                continue
            
            selected.append(candidate)
            current_sums = new_sums
        
        if len(selected) > len(best):
            best = selected
            best_sums = current_sums
    
    # Verify the best set
    if best:
        sums = all_subset_sums(best)
        if not check_divisibility(sums):
            # Fallback to a known valid set
            best = [31, 32, 34]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
