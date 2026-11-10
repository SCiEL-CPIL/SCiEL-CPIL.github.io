"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are always distinct.
    
    Uses backtracking with descending order (larger numbers first) to
    maximize set size. The key insight: placing larger numbers first
    creates wider gaps between sum layers, allowing more elements.
    """
    deadline = time.time() + 55  # safety margin before 60s limit
    
    def check_constraint(selected, candidate, n_max=256):
        """Check if adding 'candidate' to 'selected' maintains the constraint.
        selected is sorted ascending. We check that for each k, 
        max_sum(k-1) < min_sum(k)."""
        sel = sorted(selected + [candidate])
        n = len(sel)
        # For each k from 1 to n-1, check max_sum(k-1) < min_sum(k)
        # max_sum(k-1) = sum of largest (k-1) elements
        # min_sum(k) = sum of smallest k elements
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + sel[i]
        suffix = [0] * (n + 1)
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i+1] + sel[i]
        
        for k in range(1, n):
            # max sum of (k-1) elements: last (k-1) elements
            if k - 1 == 0:
                max_sum_k_minus_1 = 0
            else:
                max_sum_k_minus_1 = suffix[n - (k - 1)]
            # min sum of k elements: first k elements
            min_sum_k = prefix[k]
            if max_sum_k_minus_1 >= min_sum_k:
                return False
        return True
    
    def backtrack(selected, remaining, depth=0):
        """Backtracking search. selected is the current set (ascending),
        remaining is a list of candidates to consider (descending order)."""
        if time.time() > deadline:
            return selected
        best = list(selected)
        
        for i, cand in enumerate(remaining):
            new_remaining = remaining[i+1:]
            if check_constraint(selected, cand):
                result = backtracked_result
                new_sel = list(selected) + [cand]
                # Recurse
                prev_best = best
                sub_result = backtracking_helper(new_sel, new_remaining)
                if len(sub_result) > len(best):
                    best = sub_result
        return best
    
    def backtracking_helper(selected, remaining):
        """Recursive backtracking helper."""
        if time.time() > deadline:
            return list(selected)
        best = list(selected)
        
        for i, cand in enumerate(remaining):
            if check_constraint(selected, cand):
                new_sel = list(selected) + [cand]
                new_remaining = remaining[i+1:]
                sub_result = backtracking_helper(new_sel, new_remaining)
                if len(sub_result) > len(best):
                    best = sub_result
                # Pruning: if we've hit maximum or deadline
                if time.time() > deadline - 1:
                    break
        return best
    
    # Try starting from the full range in descending order
    # Use a smarter initial pool: focus on higher numbers for better separation
    # Start with candidates from 200 down to 1
    candidates = list(range(256, 0, -1))
    
    # Greedy warm-start: build a large valid set quickly
    greedy_set = []
    for cand in candidates:
        if check_constraint(greedy_set, cand):
            greedy_set.append(cand)
        if len(greedy_set) >= 35:
            break
    
    best_set = list(greedy_set)
    
    # Now try to improve with backtracking on subsets
    # Try removing some elements and replacing with others
    if time.time() < deadline - 5:
        # Try different starting points
        for start in range(256, 150, -5):
            if time.time() > deadline - 3:
                break
            candidates_sub = list(range(start, 0, -1))
            result = backtracking_helper([], candidates_sub)
            if len(result) > len(best_set):
                best_set = result
    
    # Final attempt with full search if time permits
    if time.time() < deadline - 2 and len(best_set) < 40:
        # Try a more targeted search
        for a_start in range(240, 180, -10):
            if time.time() > deadline - 1:
                break
            subset = list(range(a_start, 1, -1))
            result = backtracking_helper([], subset)
            if len(result) > len(best_set):
                best_set = result
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
