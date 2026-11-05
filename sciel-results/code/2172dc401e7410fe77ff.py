<<<<<<< SEARCH
def _solve_target(n):
    """Find min-k div-free smooth representation of n using DFS with pruning."""
    if n in _VAL_INDEX:
        return [n]
    
    # Candidate indices: smooth values <= n
    cand_idx = [i for i in range(_NSM) if _SMOOTH[i] <= n]
    ncand = len(cand_idx)
    if ncand == 0:
        return None
    
    # Build local incompatibility bitmask
    # Map local index to global index
    local_to_global = cand_idx
    
    # Incompatibility as bitmasks
    incompat = [0] * ncand
    for i in range(ncand):
        gi = local_to_global[i]
        for j in range(i + 1, ncand):
            gj = local_to_global[j]
            if _INCOMPAT[gi][gj]:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
    
    # Precompute suffix max sums for pruning
    # suffix_max[k][i] = max sum of k elements from cand_idx[i:]
    # This is expensive, so use a simpler bound: sum of k largest from position i onward
    # Since cand is sorted ascending, the k largest from i onward are the last k of cand[i:]
    
    # Sort candidates in descending order for better search
    cand_desc = sorted(cand_idx, key=lambda i: _SMOOTH[i], reverse=True)
    cd_vals = [_SMOOTH[i] for i in cand_desc]
    ncand = len(cd_vals)
    
    # Rebuild incompatibility for desc order
    incompat_d = [0] * ncand
    for i in range(ncand):
        gi = cand_desc[i]
        for j in range(i + 1, ncand):
            gj = cand_desc[j]
            if _INCOMPAT[gi][gj]:
                incompat_d[i] |= (1 << j)
                incompat_d[j] |= (1 << i)
    
    # Suffix sum of largest k: since sorted desc, sum of first k from position i
    # suffix_ksum[i][k] = sum of cd_vals[i:i+k]
    # Precompute for all i, k up to ncand
    suffix_sum = [[0] * (ncand + 1) for _ in range(ncand + 1)]
    for i in range(ncand):
        for k in range(1, ncand - i + 1):
            suffix_sum[i][k] = suffix_sum[i][k-1] + cd_vals[i+k-1]
    
    result = [None]
    
    def backtrack(start, k, remaining, chosen_mask, chosen_list):
        if result[0] is not None:
            return True
        if k == 0:
            if remaining == 0:
                result[0] = list(chosen_list)
                return True
            return False
        if remaining <= 0:
            return False
        if start + k > ncand:
            return False
        # Max possible sum: k largest from position start
        if suffix_sum[start][k] < remaining:
            return False
        # Min possible: at least k * 1 (all values >= 1)
        if remaining < k:
            return False
        
        for i in range(start, ncand):
            v = cd_vals[i]
            if v > remaining:
                continue
            if incompat_d[i] & chosen_mask:
                continue
            chosen_mask |= (1 << i)
            chosen_list.append(v)
            if backtrack(i + 1, k - 1, remaining - v, chosen_mask, chosen_list):
                return True
            chosen_list.pop()
            chosen_mask &= ~(1 << i)
        return False
    
    max_k = min(n, ncand)
    for k in range(2, max_k + 1):
        if time.time() - _T0 > _DEADLINE:
            break
        if backtrack(0, k, n, 0, []):
            break
    
    return result[0]
=======
def _solve_target(n):
    """Find min-k div-free smooth representation of n using optimized DFS."""
    if n in _VAL_INDEX:
        return [n]
    
    # Candidate indices: smooth values <= n
    cand_idx = [i for i in range(_NSM) if _SMOOTH[i] <= n]
    ncand = len(cand_idx)
    if ncand == 0:
        return None
    
    # Sort candidates in descending order to find large components first
    cand_desc = sorted(cand_idx, key=lambda i: _SMOOTH[i], reverse=True)
    cd_vals = [_SMOOTH[i] for i in cand_desc]
    
    # Rebuild incompatibility for desc order
    incompat_d = [0] * ncand
    for i in range(ncand):
        gi = cand_desc[i]
        for j in range(i + 1, ncand):
            gj = cand_desc[j]
            if _INCOMPAT[gi][gj]:
                incompat_d[i] |= (1 << j)
                incompat_d[j] |= (1 << i)
                
    # Precompute min suffix sums for pruning
    # min_suffix[i][k] = sum of the k smallest elements from cd_vals[i:]
    # Since cd_vals is descending, the k smallest are the last k elements of cd_vals[i:]
    # But wait, we are picking from i to end. The smallest available are at the end of the list.
    # To be safe and simple, we can just use the sum of the last k elements of the whole array as a global lower bound,
    # or better: for a specific start index, the k smallest elements in cd_vals[start:] are cd_vals[ncand-k:ncand]
    # IF start <= ncand-k. If start > ncand-k, we take cd_vals[start:ncand] and pad with 0? No, we need distinct.
    # Actually, a simple lower bound is: the sum of the k smallest elements in the entire set of candidates.
    # Let's precompute the sum of the k smallest elements in cd_vals (which are the last k elements).
    min_suffix = [[0] * (ncand + 1) for _ in range(ncand + 1)]
    for i in range(ncand):
        # The k smallest elements from i onwards are the last k elements of the subarray cd_vals[i:]
        # If we have fewer than k elements left, it's impossible.
        if ncand - i < k:
             min_suffix[i][k] = float('inf')
        else:
             # The smallest k elements in cd_vals[i:] are cd_vals[ncand-k : ncand]
             # Wait, cd_vals is sorted descending. So the smallest are at the end.
             # So the sum of the k smallest elements in cd_vals[i:] is sum(cd_vals[ncand-k:ncand])
             # This is constant for all i as long as we have at least k elements.
             # Let's precompute suffix sums of the ascending sorted list for exact min sums.
             pass
             
    # Let's use a simpler and faster pruning: 
    # Sum of the k largest elements from start onwards (Upper Bound)
    # Sum of the k smallest elements from start onwards (Lower Bound)
    
    # Precompute prefix sums for upper bound (largest k)
    # cd_vals is descending.
    # Sum of largest k from i is sum(cd_vals[i : i+k])
    prefix_sum = [0] * (ncand + 1)
    for i in range(ncand):
        prefix_sum[i+1] = prefix_sum[i] + cd_vals[i]
        
    # Precompute suffix sums for lower bound (smallest k)
    # cd_vals is descending. Smallest are at the end.
    # Sum of smallest k from i is sum(cd_vals[ncand-k : ncand]) IF i <= ncand-k.
    # Actually, if we start at i, the available elements are cd_vals[i:].
    # The smallest k of these are the last k of cd_vals[i:], which are cd_vals[ncand-k:ncand] 
    # provided i <= ncand-k. If i > ncand-k, we don't have k elements.
    # So the minimum sum of k distinct elements from cd_vals[i:] is sum(cd_vals[ncand-k:ncand])
    # Let's precompute this constant for each k.
    min_sum_k = [0] * (ncand + 1)
    for k in range(1, ncand + 1):
        min_sum_k[k] = sum(cd_vals[ncand-k:ncand])
        
    result = [None]
    
    def backtrack(start, k, remaining, chosen_mask, chosen_list):
        if result[0] is not None:
            return True
        if k == 0:
            if remaining == 0:
                result[0] = list(chosen_list)
                return True
            return False
        if remaining <= 0:
            return False
        if start + k > ncand:
            return False
            
        # Upper bound: sum of k largest from start
        max_possible = prefix_sum[start+k] - prefix_sum[start]
        if max_possible < remaining:
            return False
            
        # Lower bound: sum of k smallest available
        # The k smallest available are the last k elements of cd_vals
        # This is a valid lower bound regardless of start, as long as we have enough elements.
        if remaining < min_sum_k[k]:
            return False
            
        for i in range(start, ncand):
            v = cd_vals[i]
            if v > remaining:
                # Since sorted desc, all subsequent are smaller, so we might still find a solution
                # But if v > remaining, we can't pick v.
                # We can break if we know that even the smallest possible sum of remaining k-1 elements
                # plus the smallest available element here is too large? No.
                # We just skip this element.
                continue
            if incompat_d[i] & chosen_mask:
                continue
                
            chosen_mask |= (1 << i)
            chosen_list.append(v)
            if backtrack(i + 1, k - 1, remaining - v, chosen_mask, chosen_list):
                return True
            chosen_list.pop()
            chosen_mask &= ~(1 << i)
        return False
    
    max_k = min(n, ncand)
    for k in range(2, max_k + 1):
        if time.time() - _T0 > _DEADLINE:
            break
        if backtrack(0, k, n, 0, []):
            break
    
    return result[0]
>>>>>>> REPLACE

Now I will remove the inefficient fallback phases and replace them with a more efficient direct call to `_solve_target` for all remaining targets, ensuring we use the improved search. I'll also clean up the unused functions.

