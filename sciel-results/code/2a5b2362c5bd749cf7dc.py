"""Low discrepancy on all arithmetic progressions - Greedy Construction with Targeted Multi-Flip Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _build_ap_structure(n):
    """
    Precompute all arithmetic progressions and their constituent indices.
    Returns a list of arrays, where each array contains the indices of an AP.
    Only APs of length >= 2 are stored to save memory/time, as length 1 has fixed sum +/-1.
    """
    aps = []
    for d in range(1, n):
        max_start = n - d
        if max_start <= 0:
            continue
        for i in range(max_start):
            seq = np.arange(i, n, d)
            if len(seq) >= 2:
                aps.append(seq)
    return aps

def _compute_ap_extrema(signs, aps):
    """
    Compute the signed sum for each AP.
    Returns an array of sums.
    """
    sums = np.zeros(len(aps), dtype=np.int32)
    for idx, ap in enumerate(aps):
        sums[idx] = np.sum(signs[ap])
    return sums

def _update_single_ap(signs, ap, old_val, new_val):
    """
    Update the sum of a single AP after flipping a sign at index k.
    Returns the new sum.
    """
    # old_val is the sign before flip, new_val is after flip.
    # Actually, we need to know the index k.
    # This function is a bit abstract. Let's just recompute or use a better structure.
    # For speed, we will maintain an array of sums and update it.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    n = 256
    rng = np.random.default_rng(42)
    
    # 1. Build AP structure
    aps = _build_ap_structure(n)
    num_aps = len(aps)
    
    # Precompute for each index k, which APs contain it.
    # ap_indices[k] = list of indices in 'aps' that contain k.
    ap_indices = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for k in ap:
            ap_indices[k].append(idx)
    
    # 2. Greedy Construction
    # Initialize signs as 0 (unassigned)
    signs = np.zeros(n, dtype=np.int32)
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    
    # Order indices? Standard order 0..n-1 is fine for greedy.
    # Or maybe randomize order for robustness? Let's try standard first.
    
    for k in range(n):
        # Try +1
        # Calculate new max discrepancy if signs[k] = 1
        # We need to update ap_sums for all APs containing k.
        
        # Save original sums for these APs
        affected_aps = ap_indices[k]
        original_sums = ap_sums[affected_aps].copy()
        
        # Apply +1
        ap_sums[affected_aps] += 1
        max_disc_plus = np.max(np.abs(ap_sums))
        
        # Apply -1 (revert +1, apply -1)
        ap_sums[affected_aps] -= 2  # -1 is equivalent to -1 - 1 (since we had +1)
        # Wait, if we add 1, sum becomes S+1. If we want -1, sum should be S-1.
        # So delta is -2. Correct.
        max_disc_minus = np.max(np.abs(ap_sums))
        
        # Choose the one with lower max discrepancy
        if max_disc_plus <= max_disc_minus:
            signs[k] = 1
            # Keep the +1 state
        else:
            signs[k] = -1
            # Keep the -1 state (currently ap_sums reflects -1 state)
        
        # Note: ap_sums is now in the correct state for the chosen sign.
    
    current_disc = int(np.max(np.abs(ap_sums)))
    best_signs = signs.copy()
    best_disc = current_disc
    
    # 3. Targeted Multi-Flip Local Search
    # We will try to improve by flipping 2 or 3 signs.
    # To find good candidates, we look at the APs that are currently at the max discrepancy.
    
    # Helper to get worst APs
    def get_worst_aps(ap_sums, current_disc, num_to_get=5):
        # Find indices where |sum| == current_disc
        worst_indices = np.where(np.abs(ap_sums) == current_disc)[0]
        if len(worst_indices) == 0:
            return []
        # Sample up to num_to_get
        if len(worst_indices) > num_to_get:
            sampled = rng.choice(worst_indices, num_to_get, replace=False)
        else:
            sampled = worst_indices
        return sampled.tolist()
    
    # Main search loop
    time_limit_search = deadline - 5.0  # Leave 5s for cleanup/return
    
    while time.time() < time_limit_search:
        # Identify current worst APs
        worst_aps = get_worst_aps(ap_sums, current_disc, num_to_get=3)
        
        if not worst_aps:
            break
            
        # Collect candidate indices from these worst APs
        # We want to flip signs within these APs to reduce their sum.
        # If sum > 0, we want to flip +1 to -1.
        # If sum < 0, we want to flip -1 to +1.
        
        candidate_indices = set()
        for ap_idx in worst_aps:
            ap = aps[ap_idx]
            s = ap_sums[ap_idx]
            if s > 0:
                # Want to flip +1s to -1s
                for k in ap:
                    if signs[k] == 1:
                        candidate_indices.add(k)
            elif s < 0:
                # Want to flip -1s to +1s
                for k in ap:
                    if signs[k] == -1:
                        candidate_indices.add(k)
        
        if not candidate_indices:
            # If no obvious candidates, try random flips?
            # Or break if stuck?
            # Let's try a random single flip as a kick
            k = rng.integers(0, n)
            # Apply flip
            old_sign = signs[k]
            signs[k] = -old_sign
            # Update ap_sums
            for ap_idx in ap_indices[k]:
                ap_sums[ap_idx] += -2 * old_sign  # delta is -2 * old_sign (if old=1, delta=-2; if old=-1, delta=2)
            
            new_disc = int(np.max(np.abs(ap_sums)))
            if new_disc < current_disc:
                current_disc = new_disc
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs.copy()
            else:
                # Revert
                signs[k] = old_sign
                for ap_idx in ap_indices[k]:
                    ap_sums[ap_idx] -= -2 * old_sign
            continue
        
        # Try 2-flip and 3-flip combinations from candidates
        # Limit candidate set size to avoid combinatorial explosion
        if len(candidate_indices) > 10:
            # Sample 10 candidates
            cand_list = list(candidate_indices)
            sampled_cands = rng.choice(cand_list, 10, replace=False)
        else:
            sampled_cands = list(candidate_indices)
        
        # 2-flips
        improved = False
        for k1, k2 in combinations(sampled_cands, 2):
            # Apply flip k1, k2
            old1, old2 = signs[k1], signs[k2]
            signs[k1] = -old1
            signs[k2] = -old2
            
            # Update sums
            # k1 affects ap_indices[k1]
            # k2 affects ap_indices[k2]
            # We can just update all affected APs.
            affected = set(ap_indices[k1]) | set(ap_indices[k2])
            for ap_idx in affected:
                # Calculate delta for this AP
                delta = 0
                if k1 in aps[ap_idx]:
                    delta += -2 * old1
                if k2 in aps[ap_idx]:
                    delta += -2 * old2
                ap_sums[ap_idx] += delta
            
            new_disc = int(np.max(np.abs(ap_sums)))
            
            if new_disc < current_disc:
                current_disc = new_disc
                improved = True
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs.copy()
                    # Early exit if optimal?
                    if best_disc == 1:
                        break
            else:
                # Revert
                signs[k1] = old1
                signs[k2] = old2
                for ap_idx in affected:
                    ap_sums[ap_idx] -= delta
                    # Note: 'delta' was calculated for the specific AP in the loop above.
                    # We need to revert exactly.
                    # The delta calculation inside the loop was specific to that AP.
                    # So ap_sums[ap_idx] -= delta is correct.
            
            if time.time() > time_limit_search:
                break
            if best_disc == 1:
                break
        
        if best_disc == 1 or time.time() > time_limit_search:
            break
            
        # 3-flips (only if 2-flips didn't improve much or to escape deeper minima)
        # To save time, only do 3-flips if we are stuck or periodically.
        # Let's do a few 3-flips.
        if len(sampled_cands) >= 3:
            # Limit number of 3-flip trials
            max_3flip_trials = 10
            trials = 0
            for k1, k2, k3 in combinations(sampled_cands, 3):
                if trials >= max_3flip_trials:
                    break
                trials += 1
                
                old1, old2, old3 = signs[k1], signs[k2], signs[k3]
                signs[k1] = -old1
                signs[k2] = -old2
                signs[k3] = -old3
                
                affected = set(ap_indices[k1]) | set(ap_indices[k2]) | set(ap_indices[k3])
                for ap_idx in affected:
                    delta = 0
                    if k1 in aps[ap_idx]:
                        delta += -2 * old1
                    if k2 in aps[ap_idx]:
                        delta += -2 * old2
                    if k3 in aps[ap_idx]:
                        delta += -2 * old3
                    ap_sums[ap_idx] += delta
                
                new_disc = int(np.max(np.abs(ap_sums)))
                
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if new_disc < best_disc:
                        best_disc = new_disc
                        best_signs = signs.copy()
                        if best_disc == 1:
                            break
                else:
                    # Revert
                    signs[k1] = old1
                    signs[k2] = old2
                    signs[k3] = old3
                    for ap_idx in affected:
                        ap_sums[ap_idx] -= delta
                
                if time.time() > time_limit_search:
                    break
                if best_disc == 1:
                    break
            
            if best_disc == 1:
                break

    return best_signs.tolist()

# EVOLVE-BLOCK-END
