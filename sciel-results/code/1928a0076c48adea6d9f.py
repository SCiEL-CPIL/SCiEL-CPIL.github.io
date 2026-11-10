"""
Strategy: Iterated Local Search (ILS) with Direct Max-Discrepancy Minimization.

The previous approach used a smooth proxy (sum of squares) which is a weak surrogate for the L-infinity norm (max). 
This approach directly optimizes the max discrepancy by:
1. Pre-computing all arithmetic progressions (APs) and their index mappings.
2. Maintaining the current prefix sums for each AP.
3. For each potential flip, calculating the change in the max discrepancy for the affected APs only.
4. Using a greedy hill-climbing phase followed by random perturbations (ILS) to escape local optima.
5. Vectorizing the discrepancy check to allow many more iterations within the time budget.

Key Insight: The number of APs containing any specific index $i$ is relatively small compared to the total number of APs. 
Thus, the cost of evaluating a flip is $O(\text{degree}(i))$, which is much lower than $O(\text{Total APs})$.
"""

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_ap_structure(n):
    """
    Pre-compute all arithmetic progressions and map indices to the progressions they belong to.
    An AP is defined by (start, step). It includes all elements start, start+step, ... < n.
    The discrepancy for an AP is the max absolute value of its prefix sums.
    
    Returns:
    - ap_indices: List of numpy arrays of indices for each AP.
    - index_to_aps: List of lists of (ap_id, position_in_ap) for each index 0..n-1.
    """
    ap_indices = []
    index_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    # Step can be from 1 to n. 
    # Start can be from 0 to n-1.
    # But if start >= n, no elements.
    # If step > n, only 1 element (the start itself).
    
    for step in range(1, n + 1):
        for start in range(n):
            if start >= n:
                break
            indices = np.arange(start, n, step, dtype=np.int32)
            if len(indices) == 0:
                continue
            
            ap_indices.append(indices)
            
            # Map each index in this AP to (ap_id, position)
            for pos, idx in enumerate(indices):
                index_to_aps[idx].append((ap_id, pos))
            
            ap_id += 1
            
    return ap_indices, index_to_aps

def _init_prefix_sums(signs, ap_indices):
    """
    Compute initial prefix sums for all APs.
    Returns a list of numpy arrays, where prefix_sums[ap_id] is the array of prefix sums.
    """
    prefix_sums = []
    signs_arr = np.array(signs, dtype=np.int64)
    for indices in ap_indices:
        subseq = signs_arr[indices]
        ps = np.cumsum(subseq)
        prefix_sums.append(ps)
    return prefix_sums

def _get_max_disc(prefix_sums):
    """
    Compute the global max discrepancy from the list of prefix sums.
    max_disc = max( max(abs(ps)) for ps in prefix_sums )
    """
    # Vectorized max over all prefix sums
    # prefix_sums is a list of arrays of varying lengths.
    # We can't easily stack them if lengths differ, but we can just loop.
    # However, len(prefix_sums) is ~ n^2/2 = 32k for n=256.
    # A Python loop over 32k items taking np.max is fast (~ms).
    max_disc = 0
    for ps in prefix_sums:
        m = np.max(np.abs(ps))
        if m > max_disc:
            max_disc = m
    return max_disc

def _eval_flip(signs, i, index_to_aps, prefix_sums):
    """
    Evaluate the change in max discrepancy if we flip sign at index i.
    Returns the new max discrepancy.
    This is a dry-run (does not modify state).
    """
    old_val = signs[i]
    delta = -2 * old_val # new_val - old_val = -old - old = -2*old
    
    # We need to check if any AP's max abs prefix sum exceeds the current global max
    # Actually, we just need to compute the new global max.
    # To be fast, we can first check if the flip *can* reduce the max.
    # If the current max is D, and the APs that *have* a prefix sum of +/- D are not affected by the flip, 
    # then the max remains D (or decreases if we break the tie, but we can't know without checking).
    # However, if the flip *increases* the max in any affected AP beyond D, the new max is > D.
    
    # Simplest robust method:
    # 1. Identify affected APs.
    # 2. For each affected AP, compute the new max abs prefix sum.
    # 3. The new global max is max( max_new_affected, max_unaffected ).
    # We don't have max_unaffected stored explicitly, but we know the old global max.
    # If no affected AP has a new max > old_global_max, then new_global_max <= old_global_max.
    # Specifically, new_global_max = max( max( new_max_ap for affected ), max( old_max_ap for unaffected ) ).
    # Since max( old_max_ap for unaffected ) <= old_global_max, 
    # if max( new_max_ap for affected ) < old_global_max, then new_global_max < old_global_max is possible?
    # No, if the AP that had the old global max is NOT affected, then new_global_max = old_global_max.
    # If the AP that had the old global max IS affected, we must recompute its max.
    
    # Optimization:
    # We can just recompute the max for affected APs.
    # Let M_aff = max( new_max_abs_ps(ap) for ap in affected )
    # Let M_unaff = max( old_max_abs_ps(ap) for ap in unaffected )
    # We don't store old_max_abs_ps for each AP. 
    # However, if we maintain an array `ap_max_disc` for each AP, we can update it.
    
    # Let's assume we have `ap_max_disc` array.
    # This function is a dry run, so it can't easily update the persistent `ap_max_disc` without side effects.
    # So we will just return the new max for the affected APs.
    # The caller (construct) will handle the logic.
    
    max_new_aff = 0
    for ap_id, pos in index_to_aps[i]:
        ps = prefix_sums[ap_id]
        # The suffix from pos to end changes by delta
        # new_ps_suffix = ps[pos:] + delta
        # We need max(abs(ps[:pos]), abs(ps[pos:] + delta))
        # ps[:pos] is unchanged.
        # max_old_prefix = max(abs(ps[:pos])) if pos > 0 else 0
        # max_new_suffix = max(abs(ps[pos:] + delta))
        
        # Vectorized check
        # We can create a temporary array or just compute max
        # Since lengths are small, a simple loop or vectorized op is fine.
        
        # Optimization: 
        # If pos == 0, the whole array changes.
        # If pos > 0, the prefix part is unchanged.
        
        # To avoid copying, we can compute:
        # max_abs_old_prefix = np.max(np.abs(ps[:pos])) if pos > 0 else 0
        # max_abs_new_suffix = np.max(np.abs(ps[pos:] + delta))
        
        # But we need the MAX of these two to be the new max for this AP.
        
        # Let's just compute the max of the modified suffix and the unchanged prefix.
        if pos > 0:
            max_prefix_part = np.max(np.abs(ps[:pos]))
        else:
            max_prefix_part = 0
            
        # The suffix part
        # We don't want to allocate a new array if possible, but ps[pos:] + delta creates a view/copy.
        # For n=256, this is tiny.
        max_suffix_part = np.max(np.abs(ps[pos:] + delta))
        
        new_ap_max = max(max_prefix_part, max_suffix_part)
        if new_ap_max > max_new_aff:
            max_new_aff = new_ap_max
            
    return max_new_aff

def construct():
    n = 256
    deadline = time.time() + 55  # Safety margin
    
    # Pre-compute structure
    ap_indices, index_to_aps = _build_ap_structure(n)
    num_aps = len(ap_indices)
    
    # We will maintain:
    # signs: list of int
    # prefix_sums: list of np arrays
    # ap_max_disc: np array of shape (num_aps,), storing max(abs(prefix_sums[ap_id]))
    # global_max_disc: int
    
    best_signs = None
    best_global_max = float('inf')
    
    rng = random.Random(123)
    
    # ILS Loop
    while time.time() < deadline:
        # Initialize random signs
        signs = [rng.choice((-1, 1)) for _ in range(n)]
        prefix_sums = _init_prefix_sums(signs, ap_indices)
        
        # Initialize ap_max_disc
        ap_max_disc = np.array([np.max(np.abs(ps)) for ps in prefix_sums], dtype=np.int64)
        global_max_disc = np.max(ap_max_disc)
        
        # Local Search Phase (Greedy Hill Climbing)
        # We try to reduce global_max_disc.
        # We will iterate through indices. If flipping an index reduces global_max_disc, we accept.
        # To speed up, we can stop early if we reach a theoretical lower bound (e.g., 1 or 2).
        # Lower bound is 1 (singletons).
        
        max_ls_iters = 5000
        ls_iter = 0
        stuck_count = 0
        current_global_max = global_max_disc
        
        while ls_iter < max_ls_iters and time.time() < deadline - 5:
            improved = False
            # Shuffle indices to avoid bias
            indices_to_try = list(range(n))
            rng.shuffle(indices_to_try)
            
            for i in indices_to_try:
                if time.time() > deadline - 5:
                    break
                    
                # Evaluate flip at i
                old_val = signs[i]
                delta = -2 * old_val
                
                # Find max new disc for affected APs
                max_new_aff = 0
                # Also track if the AP that currently holds the global max is affected
                # Actually, we just need to know the new global max.
                # New Global Max = max( max_new_aff, max_unaffected )
                # max_unaffected is the max of ap_max_disc for APs NOT in index_to_aps[i].
                # This is hard to compute quickly without a data structure.
                
                # Alternative: 
                # If max_new_aff < current_global_max, it's possible the new global max is lower.
                # But if the AP with current_global_max is NOT affected, new global max = current_global_max.
                # So improvement is ONLY possible if ALL APs with value == current_global_max are affected 
                # AND their new max is < current_global_max.
                
                # Let's identify affected APs
                affected_aps = index_to_aps[i]
                
                # Check if any AP with max == current_global_max is NOT in affected_aps
                # This check is O(num_aps) which is too slow inside the inner loop.
                
                # Heuristic: 
                # Just compute the new max for affected APs.
                # If max_new_aff < current_global_max, we *might* have improved.
                # We must verify by checking if any *unaffected* AP has max == current_global_max.
                
                # To make this efficient, we can maintain a count of how many APs have max == current_global_max.
                # But current_global_max changes.
                
                # Let's use a simpler, slightly slower but robust check for the "potential improvement":
                # 1. Compute max_new_aff.
                # 2. If max_new_aff >= current_global_max: No improvement possible (max can only stay same or increase).
                # 3. If max_new_aff < current_global_max: 
                #    We need to check if there is any AP NOT in affected_aps with ap_max_disc == current_global_max.
                #    If yes, new global max = current_global_max (no improvement).
                #    If no, new global max = max(max_new_aff, second_max_unaffected).
                #    This requires knowing the second max.
                
                # Given the complexity, let's stick to a simpler heuristic that is still effective:
                # We accept a flip if it reduces the max of the affected APs AND 
                # we assume it helps the global max. 
                # Actually, if we just greedily minimize the max of the *affected* APs, we often reduce the global max.
                
                # Let's compute max_new_aff.
                for ap_id, pos in affected_aps:
                    ps = prefix_sums[ap_id]
                    if pos > 0:
                        max_pref = np.max(np.abs(ps[:pos]))
                    else:
                        max_pref = 0
                    max_suff = np.max(np.abs(ps[pos:] + delta))
                    new_ap_max = max(max_pref, max_suff)
                    if new_ap_max > max_new_aff:
                        max_new_aff = new_ap_max
                        
                # If max_new_aff is less than current_global_max, it's a candidate.
                # We must ensure we don't increase the global max.
                # If max_new_aff > current_global_max, reject.
                if max_new_aff > current_global_max:
                    continue
                    
                # If max_new_aff == current_global_max, it might be neutral or bad. 
                # We only accept if it strictly reduces the global max.
                # To check if it strictly reduces, we need to know if any OTHER AP (unaffected) has max == current_global_max.
                # This is expensive. 
                
                # Strategy: 
                # If max_new_aff < current_global_max, we tentatively accept.
                # We then recompute the global max (O(num_aps) ~ 32k ops). 
                # This is only done for promising moves.
                
                # Apply flip tentatively
                signs[i] = -old_val
                for ap_id, pos in affected_aps:
                    prefix_sums[ap_id][pos:] += delta
                    # Update ap_max_disc for this AP
                    new_m = np.max(np.abs(prefix_sums[ap_id]))
                    ap_max_disc[ap_id] = new_m
                
                new_global_max = np.max(ap_max_disc)
                
                if new_global_max < current_global_max:
                    current_global_max = new_global_max
                    improved = True
                    stuck_count = 0
                else:
                    # Revert
                    signs[i] = old_val
                    for ap_id, pos in affected_aps:
                        prefix_sums[ap_id][pos:] -= delta
                        ap_max_disc[ap_id] = np.max(np.abs(prefix_sums[ap_id]))
                    stuck_count += 1
                    
            ls_iter += 1
            if not improved:
                break
                
        # End Local Search
        
        # Update best
        if current_global_max < best_global_max:
            best_global_max = current_global_max
            best_signs = list(signs)
            
        # Perturbation for ILS
        # Flip a few random signs to escape local optimum
        num_flips = rng.randint(3, 8)
        for _ in range(num_flips):
            i = rng.randint(0, n - 1)
            signs[i] = -signs[i]
            delta = -2 * signs[i] # Wait, we flipped it, so delta for the *next* flip logic?
            # No, we just need to update the state structures for the perturbation.
            # The signs list is updated. We need to update prefix_sums and ap_max_disc.
            
            # This perturbation update is O(degree(i))
            old_val = signs[i] # This is the NEW value
            # We flipped from -old_val to old_val.
            # The change in value is 2*old_val.
            delta_pert = 2 * old_val
            for ap_id, pos in index_to_aps[i]:
                prefix_sums[ap_id][pos:] += delta_pert
                ap_max_disc[ap_id] = np.max(np.abs(prefix_sums[ap_id]))
                
            # Note: We don't update current_global_max here, it will be recalculated in the next LS phase.
            # But we should probably update it to avoid confusion, though LS starts by checking moves.
            # Actually, the LS phase assumes `current_global_max` is correct.
            # So we must recalculate it after perturbation.
            
        # Recalculate global max for next LS phase
        global_max_disc = np.max(ap_max_disc)

    if best_signs is None:
        best_signs = [1] * n
        
    return best_signs
# EVOLVE-BLOCK-END
