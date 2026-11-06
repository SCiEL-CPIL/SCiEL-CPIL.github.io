import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    max_target = 128
    
    # 1. Generate smooth numbers 2^a * 3^b * 5^c <= max_target
    smooth = []
    for a in range(0, 8):  # 2^7 = 128
        v2 = 2 ** a
        if v2 > max_target: break
        for b in range(0, 6): # 3^5 = 243
            v3 = 3 ** b
            if v2 * v3 > max_target: break
            for c in range(0, 4): # 5^3 = 125
                v5 = 5 ** c
                val = v2 * v3 * v5
                if val <= max_target:
                    smooth.append(val)
                else:
                    break
    smooth = sorted(set(smooth))
    N_smooth = len(smooth)
    
    # 2. Precompute divisibility matrix
    # divides[i][j] is True if smooth[i] divides smooth[j]
    divides = [[False] * N_smooth for _ in range(N_smooth)]
    for i in range(N_smooth):
        for j in range(N_smooth):
            if smooth[j] % smooth[i] == 0:
                divides[i][j] = True
            else:
                # Optimization: if smooth[i] > smooth[j], it can't divide unless equal, 
                # but distinctness is handled by indices. 
                # Actually, if smooth[i] > smooth[j], smooth[i] cannot divide smooth[j] unless they are equal, 
                # but we are picking distinct terms. 
                # Wait, the condition is "no chosen term may divide another".
                # So if we pick i and j, we need NOT (smooth[i] % smooth[j] == 0) AND NOT (smooth[j] % smooth[i] == 0).
                pass
    
    # Helper to check if a new term index 'new_idx' is compatible with a list of selected indices 'selected'
    def is_compatible(new_idx, selected):
        for idx in selected:
            if divides[new_idx][idx] or divides[idx][new_idx]:
                return False
        return True

    # 3. Solve for each target
    answers = []
    
    # Pre-filter smooth numbers for each target to speed up search
    # candidates[n] = list of indices of smooth numbers <= n
    candidates = [[] for _ in range(max_target + 1)]
    for i, val in enumerate(smooth):
        for n in range(val, max_target + 1):
            candidates[n].append(i)
    
    for n in range(1, max_target + 1):
        if time.time() > deadline:
            # Emergency fallback: return whatever we have, or a valid but suboptimal solution
            # To ensure validity, we must have a valid solution. 
            # If we run out of time, we might have incomplete answers.
            # Let's hope we don't. The search is very fast for these sizes.
            pass
            
        found = None
        cands = candidates[n]
        
        # Try k from 1 to len(cands)
        # In practice, k is small (1, 2, 3, 4)
        for k in range(1, len(cands) + 1):
            if time.time() > deadline:
                break
            found = search_minimal_antichain(n, k, cands, smooth, divides)
            if found is not None:
                break
        
        if found is None:
            # Fallback: should not happen for 1..128
            # Use a simple greedy or return [n] if n is smooth and antichain (trivially)
            # If n is smooth, [n] is a valid antichain.
            if n in smooth:
                found = [n]
            else:
                # This case is extremely unlikely to fail for 1..128
                # We can use a slow brute force as last resort
                found = brute_force_fallback(n, smooth, divides)
                
        answers.append(found)
        
    return answers

def search_minimal_antichain(target, k, candidate_indices, smooth, divides):
    """
    Finds a subset of 'k' indices from 'candidate_indices' such that:
    1. Sum of corresponding smooth numbers == target
    2. The set forms an antichain.
    Returns list of values or None.
    """
    # Sort candidates by value descending to help with pruning (sum bounds)
    # But we need to maintain indices for divisibility check
    # Let's just use the indices as is, but sort them to have a consistent order for backtracking
    # We need to avoid duplicates in the combination generation, so we pick indices in increasing order.
    
    # Pre-calculate suffix sums of the largest possible values to prune?
    # Since k is small, simple backtracking is fine.
    
    selected_indices = []
    
    def backtrack(start_idx, current_sum, current_count):
        if current_count == k:
            if current_sum == target:
                return [smooth[i] for i in selected_indices]
            return None
        
        # Pruning: if even adding the largest remaining (k - current_count) terms 
        # from candidate_indices[start_idx:] doesn't reach target, prune.
        # Since candidate_indices are not sorted by value, this bound is hard to calculate quickly.
        # However, we can just check if current_sum > target.
        if current_sum > target:
            return None
            
        # If current_sum + (target - current_sum) < target? No.
        # If current_sum + sum of top (k-current_count) values in candidates[start_idx:] < target, prune.
        # Let's implement a simple bound: max possible add is sum of largest (k-current_count) values 
        # from the remaining candidates.
        remaining_needed = k - current_count
        if remaining_needed > 0:
            # Get values of candidates from start_idx onwards
            # This is O(N) per step, might be slow if N is large, but N is small (< 50)
            rem_vals = [smooth[idx] for idx in candidate_indices[start_idx:]]
            rem_vals.sort(reverse=True)
            max_possible_add = sum(rem_vals[:remaining_needed])
            if current_sum + max_possible_add < target:
                return None
        
        for i in range(start_idx, len(candidate_indices)):
            idx = candidate_indices[i]
            val = smooth[idx]
            
            # If adding this val exceeds target, and values are not sorted, we can't break.
            # But we can skip if val > target - current_sum.
            if current_sum + val > target:
                continue
                
            # Check compatibility with selected
            if not is_compatible(idx, selected_indices):
                continue
                
            selected_indices.append(idx)
            res = backtrack(i + 1, current_sum + val, current_count + 1)
            if res is not None:
                return res
            selected_indices.pop()
            
        return None

    return backtrack(0, 0, 0)

def brute_force_fallback(n, smooth, divides):
    # Extremely slow fallback, just to ensure we return something valid if search fails
    # Try all subsets of size 1 to 10
    indices = [i for i, v in enumerate(smooth) if v <= n]
    for k in range(1, 11):
        for combo in combinations(indices, k):
            if sum(smooth[i] for i in combo) == n:
                # Check antichain
                valid = True
                for a in range(len(combo)):
                    for b in range(a+1, len(combo)):
                        if divides[combo[a]][combo[b]] or divides[combo[b]][combo[a]]:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    return [smooth[i] for i in combo]
    # If still not found, return [n] if possible, else [] (invalid)
    # For 1..128, this should never be reached
    return [n] if n in smooth else []
