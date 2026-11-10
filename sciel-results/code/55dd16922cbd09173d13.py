from fractions import Fraction
import random


def _can_make_subset_sum(target, recips):
    """Check if target can be formed by a subset sum of recips.
    recips should be sorted in descending order for pruning efficiency.
    """
    n = len(recips)
    if n == 0:
        return target == 0
    
    # Precompute suffix sums for pruning
    suffix_sum = [Fraction(0, 1)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recips[i]

    def dfs(idx, current_target):
        if current_target == 0:
            return True
        if idx == n or current_target < 0:
            return False
        # If even taking all remaining can't reach target, prune
        if suffix_sum[idx] < current_target:
            return False
        
        # Try including recips[idx]
        if dfs(idx + 1, current_target - recips[idx]):
            return True
        
        # Try excluding recips[idx]
        if dfs(idx + 1, current_target):
            return True
            
        return False

    return dfs(0, target)


def _is_subsum_free(selected, candidate):
    """Check that no reciprocal of a selected element equals the sum of
    reciprocals of a nonempty subset of the OTHER selected elements.
    'candidate' is the element to add; we verify the full set
    selected + [candidate].
    """
    S = selected + [candidate]
    n = len(S)
    recip = {x: Fraction(1, x) for x in S}

    for i in range(n):
        target = recip[S[i]]
        others = [S[j] for j in range(n) if j != i]
        # Get reciprocals of others
        others_recips = [recip[x] for x in others]
        # Sort descending to help pruning
        others_recips.sort(reverse=True)
        
        # Check if target is a subset sum of others_recips
        # Note: subset must be non-empty. 
        # If target > 0, and sum of empty set is 0, we are looking for non-empty.
        # _can_make_subset_sum returns True if target is reachable.
        # Since target > 0, the empty set (sum 0) won't match unless target==0, which is not the case.
        if _can_make_subset_sum(target, others_recips):
            return False
    return True


def construct():
    best_set = []
    
    # Define several orders to try
    orders = []
    orders.append(list(range(24, 0, -1))) # Descending
    orders.append(list(range(1, 25)))     # Ascending
    
    # Add random permutations
    for _ in range(100):
        perm = list(range(1, 25))
        random.shuffle(perm)
        orders.append(perm)
        
    for order in orders:
        selected = []
        for cand in order:
            if _is_subsum_free(selected, cand):
                selected.append(cand)
        if len(selected) > len(best_set):
            best_set = selected
            
    # Fallback: guarantee a valid nonempty set
    if not best_set:
        return [1]
    return list(best_set)
