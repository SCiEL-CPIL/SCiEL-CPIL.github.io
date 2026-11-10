"""Program for the suite task: maximize cardinality of S ⊆ [1,100] with no subset summing to 200."""
# EVOLVE-BLOCK-START
def construct():
    # Structural insight: 
    # If S = {a, a+1, ..., 100}, then:
    #   max 2-sum = (99+100) = 199 < 200
    #   min 3-sum = a + (a+1) + (a+2) = 3a+3
    # If 3a+3 > 200, i.e., a >= 66, then no subset sums to 200.
    # Base: {66, 67, ..., 100} has 35 elements.
    
    # Try to improve by adding smaller elements and removing minimal blockers.
    # We'll use a smart greedy + local search approach.
    
    from itertools import combinations
    
    # Start with the interval solution
    S = set(range(66, 101))
    
    # Function to check if a subset of S sums to 200
    # Use meet-in-the-middle for efficiency
    def has_subset_sum(target, elements):
        elements = sorted(elements)
        n = len(elements)
        if n <= 20:
            # Direct check using bitset or DP
            reachable = {0}
            for x in elements:
                new = {s + x for s in reachable if s + x <= target}
                if target in new:
                    return True
                reachable |= new
            return target in reachable
        else:
            # Meet in the middle
            half = n // 2
            left = elements[:half]
            right = elements[half:]
            
            left_sums = {0}
            for x in left:
                new = {s + x for s in left_sums if s + x <= target}
                left_sums |= new
            
            right_sums = {0}
            for x in right:
                new = {s + x for s in right_sums if s + x <= target}
                right_sums |= new
            
            for s in left_sums:
                if target - s in right_sums:
                    return True
            return False
    
    # Verify base solution
    assert not has_subset_sum(200, list(S)), "Base solution is invalid!"
    
    # Try to add elements from below 66, one by one, checking validity
    # For each candidate x < 66, check if S ∪ {x} is valid
    # If not, try removing some elements from S to make it valid
    # We want to maximize |S|, so we try to add as many as possible while removing few
    
    added = 0
    for x in range(65, 0, -1):
        # Try adding x
        trial = S | {x}
        if not has_subset_sum(200, list(trial)):
            S = trial
            added += 1
        else:
            # Need to remove some elements to make it valid
            # Find minimal set of elements to remove from S (not x) such that S\removed ∪ {x} is valid
            # Try removing 1 element, then 2, etc.
            s_list = list(S - {x})  # elements currently in S (x not yet added)
            
            found = False
            for r in range(1, min(len(s_list), 4) + 1):  # Try removing up to 3 elements
                for removal in combinations(s_list, r):
                    candidate = S - set(removal)
                    candidate.add(x)
                    if not has_subset_sum(200, list(candidate)):
                        S = candidate
                        added += 1 - r  # Net change in size
                        found = True
                        break
                if found:
                    break
            
            # If we couldn't make it work with up to 3 removals, skip x
            # (adding x would cost more than it gains)
    
    # Final verification
    assert not has_subset_sum(200, list(S)), "Final solution is invalid!"
    assert len(S) > 0, "Solution must be nonempty"
    
    return list(S)
# EVOLVE-BLOCK-END
