"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to leave 10s safety margin
    
    n = 256
    
    # Precompute all arithmetic progressions (start, step, length)
    # An AP is defined by (start, step, length) where:
    #   indices are: start, start+step, start+2*step, ..., start+(length-1)*step
    #   all indices must be in [0, n-1] (0-indexed) or [1, n] (1-indexed)
    # Let's use 0-indexed internally.
    
    # Build incidence: for each position i, which APs contain it?
    # Actually, we need: for each AP, the list of positions it covers.
    # Then a flip at position i affects all APs that contain i.
    
    # First, enumerate all APs.
    aps = []  # list of tuples (list of positions)
    for step in range(1, n + 1):
        for start in range(0, n):
            # Generate the AP: start, start+step, start+2*step, ... < n
            positions = list(range(start, n, step))
            if len(positions) >= 1:
                aps.append(positions)
    
    num_aps = len(aps)
    
    # Build reverse incidence: for each position, which AP indices contain it?
    pos_to_aps = [[] for _ in range(n)]
    for ap_idx, positions in enumerate(aps):
        for pos in positions:
            pos_to_aps[pos].append(ap_idx)
    
    # Compute initial AP sums for a random starting assignment
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=n)
    
    # Compute all AP sums
    ap_sums = np.zeros(num_aps, dtype=np.int64)
    for ap_idx, positions in enumerate(aps):
        ap_sums[ap_idx] = np.sum(signs[positions])
    
    max_abs = int(np.max(np.abs(ap_sums)))
    
    # Local search: flip a single sign if it reduces the max absolute AP sum
    # Use a histogram of |ap_sums| to quickly find the max
    # And track which positions are in which APs for efficient updates
    
    best_signs = signs.copy()
    best_max_abs = max_abs
    
    # For efficiency, we'll track:
    # - ap_sums: current sum for each AP
    # - For each position, the list of APs it belongs to (precomputed)
    
    # We'll use a simple approach: try flipping each position, check if max decreases
    # To speed up, we can track the current max and only consider positions
    # that are in APs achieving the current max.
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Find positions that are in APs with |sum| == max_abs
        # Flip one such position and see if it helps
        
        # Get all APs that currently have max absolute sum
        max_mask = np.abs(ap_sums) == max_abs
        if not np.any(max_mask):
            break
        
        # Collect candidate positions from these APs
        candidate_positions = set()
        for ap_idx in np.where(max_mask)[0]:
            for pos in aps[ap_idx]:
                candidate_positions.add(pos)
        
        if not candidate_positions:
            break
        
        # Try flipping each candidate position
        improved = False
        # Shuffle candidates for diversity
        cand_list = list(candidate_positions)
        rng.shuffle(cand_list)
        
        for pos in cand_list:
            if time.time() > deadline:
                break
            
            # Try flipping position pos
            old_sign = signs[pos]
            new_sign = -old_sign
            signs[pos] = new_sign
            
            # Update affected AP sums
            affected_aps = pos_to_aps[pos]
            old_max = max_abs
            new_max = 0
            
            # We need to recompute max. For efficiency, let's just update the affected APs
            # and track the new max.
            # Since only affected APs change, the new max is either:
            # - the max of unaffected APs (which was <= old_max, and if old_max APs are affected, could decrease)
            # - the max of affected APs after update
            
            # Simple approach: update affected APs, then scan all APs for new max.
            # This is O(num_aps) per flip attempt, which is ~10k. With 256 positions to try,
            # that's ~2.5M ops per outer iteration. Might be too slow.
            
            # Better: track a histogram of |ap_sums| values.
            # Actually, let's just do it and see if it's fast enough.
            
            for ap_idx in affected_aps:
                ap_sums[ap_idx] += 2 * new_sign  # sign changed from old to new, delta is 2*new_sign... 
                # Wait: old contribution was signs[pos] * 1 = old_sign. New is new_sign.
                # Delta = new_sign - old_sign = new_sign - (-new_sign) = 2*new_sign.
                # So ap_sums[ap_idx] += 2 * new_sign.
            
            # Find new max
            new_max = int(np.max(np.abs(ap_sums)))
            
            if new_max < old_max:
                # Keep the flip
                max_abs = new_max
                improved = True
                if max_abs < best_max_abs:
                    best_max_abs = max_abs
                    best_signs = signs.copy()
                break
            else:
                # Revert
                signs[pos] = old_sign
                for ap_idx in affected_aps:
                    ap_sums[ap_idx] -= 2 * new_sign  # revert
        
        if not improved:
            # Stuck in local minimum. Try a random restart or perturbation.
            # Perturbation: flip a random set of positions
            num_flips = rng.integers(5, 20)
            flip_positions = rng.choice(n, size=num_flips, replace=False)
            for pos in flip_positions:
                signs[pos] = -signs[pos]
                for ap_idx in pos_to_aps[pos]:
                    ap_sums[ap_idx] += 2 * signs[pos]
            max_abs = int(np.max(np.abs(ap_sums)))
            if max_abs < best_max_abs:
                best_max_abs = max_abs
                best_signs = signs.copy()
            
            # If perturbation made things worse, continue local search anyway
    
    return list(best_signs.astype(int))
# EVOLVE-BLOCK-END
