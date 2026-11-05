import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 256

    # Build all APs with length >= 2
    # An AP is defined by start index r and step d.
    # Indices: r, r+d, r+2d, ... < n
    aps = []
    ap_membership = [[] for _ in range(n)]
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)
    
    num_aps = len(aps)

    def _compute_ap_sums(signs):
        """Compute max and min subarray sum for each AP."""
        max_sums = np.zeros(num_aps, dtype=np.int32)
        min_sums = np.zeros(num_aps, dtype=np.int32)
        for i in range(num_aps):
            ap = aps[i]
            vals = np.array([signs[j] for j in ap], dtype=np.int32)
            cum = np.cumsum(vals)
            cum_ext = np.concatenate(([0], cum))
            min_prefix = np.minimum.accumulate(cum_ext)
            max_prefix = np.maximum.accumulate(cum_ext)
            # Max subarray sum: max(cum[j] - min_prefix[j])
            max_sums[i] = np.max(cum_ext - min_prefix)
            # Min subarray sum: min(cum[j] - max_prefix[j])
            min_sums[i] = np.min(cum_ext - max_prefix)
        return max_sums, min_sums

    def _get_disc(max_sums, min_sums):
        if num_aps == 0:
            return 1
        return max(int(np.max(max_sums)), -int(np.min(min_sums)), 1)

    def _update_ap(ap_idx, signs):
        """Recompute sums for a single AP."""
        ap = aps[ap_idx]
        vals = np.array([signs[j] for j in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        max_sub = int(np.max(cum_ext - min_prefix))
        min_sub = int(np.min(cum_ext - max_prefix))
        return max_sub, min_sub

    def _apply_flip(signs, flip_set, max_sums, min_sums):
        """Apply a set of flips and update AP sums incrementally."""
        # We need to be careful with incremental updates if flip_set has multiple indices.
        # It's safer to just recompute affected APs.
        # Identify unique APs affected
        affected_aps = set()
        for idx in flip_set:
            for ap_idx in ap_membership[idx]:
                affected_aps.add(ap_idx)
        
        # Flip signs
        for idx in flip_set:
            signs[idx] = -signs[idx]
            
        # Update sums for affected APs
        for ap_idx in affected_aps:
            max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
            
        return _get_disc(max_sums, min_sums)

    def _revert_flip(signs, flip_set, max_sums, min_sums):
        """Revert a set of flips and update AP sums incrementally."""
        affected_aps = set()
        for idx in flip_set:
            for ap_idx in ap_membership[idx]:
                affected_aps.add(ap_idx)
                
        for idx in flip_set:
            signs[idx] = -signs[idx]
            
        for ap_idx in affected_aps:
            max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
            
        return _get_disc(max_sums, min_sums)

    # --- Phase 1: Generate diverse candidates ---
    candidates = []

    # Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)

    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(alt)

    # Block pattern
    direct = [0] * n
    for i in range(n):
        block = (i // 4) % 2
        pos = i % 4
        if block == 0:
            direct[i] = 1 if pos < 2 else -1
        else:
            direct[i] = -1 if pos < 2 else 1
    candidates.append(direct)

    # Random
    for s in range(50):
        r = random.Random(s)
        candidates.append([r.choice((-1, 1)) for _ in range(n)])

    # --- Phase 2: Evaluate candidates, pick best ---
    best_signs = candidates[0][:]
    best_disc = float('inf')

    for cand in candidates:
        if time.time() > deadline - 30:
            break
        max_sums, min_sums = _compute_ap_sums(cand)
        d = _get_disc(max_sums, min_sums)
        if d < best_disc:
            best_disc = d
            best_signs = cand[:]

    # --- Phase 3: Multi-point Targeted Local Search ---
    signs = best_signs[:]
    max_sums, min_sums = _compute_ap_sums(signs)
    current_disc = _get_disc(max_sums, min_sums)
    
    rng = random.Random(42)
    no_improve = 0
    max_no_improve = 1000
    
    # Precompute indices of APs for quick access to "worst" APs
    # We will sample from worst APs to pick flip candidates
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Identify worst APs
        # abs_disc for each AP
        ap_abs_disc = np.maximum(max_sums, -min_sums)
        
        # Find indices of top K worst APs
        K_worst = 10
        if num_aps > K_worst:
            # Get top K indices
            # argsort is O(N log N), N is ~3000, so it's fast
            worst_indices = np.argsort(-ap_abs_disc)[:K_worst]
        else:
            worst_indices = np.arange(num_aps)
            
        # Build a pool of candidate indices to flip
        # Mix of indices from worst APs and random indices
        candidate_pool = set()
        
        # Add indices from worst APs
        for ap_idx in worst_indices:
            ap = aps[ap_idx]
            # Pick a few random indices from this AP
            for _ in range(3):
                if ap:
                    candidate_pool.add(rng.choice(ap))
        
        # Add some global random indices
        for _ in range(5):
            candidate_pool.add(rng.randint(0, n - 1))
            
        candidate_list = list(candidate_pool)
        
        # Try different flip set sizes: 1, 2, 3
        found_improvement = False
        
        # Try size 1
        if len(candidate_list) > 0:
            idx = rng.choice(candidate_list)
            flip_set = [idx]
            new_disc = _apply_flip(signs, flip_set, max_sums, min_sums)
            if new_disc < current_disc:
                current_disc = new_disc
                no_improve = 0
                found_improvement = True
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                _revert_flip(signs, flip_set, max_sums, min_sums)
                no_improve += 1
        
        # Try size 2
        if not found_improvement and len(candidate_list) >= 2:
            # Pick 2 distinct indices
            idx1, idx2 = rng.sample(candidate_list, 2)
            flip_set = [idx1, idx2]
            new_disc = _apply_flip(signs, flip_set, max_sums, min_sums)
            if new_disc < current_disc:
                current_disc = new_disc
                no_improve = 0
                found_improvement = True
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                _revert_flip(signs, flip_set, max_sums, min_sums)
                no_improve += 1
                
        # Try size 3
        if not found_improvement and len(candidate_list) >= 3:
            idx1, idx2, idx3 = rng.sample(candidate_list, 3)
            flip_set = [idx1, idx2, idx3]
            new_disc = _apply_flip(signs, flip_set, max_sums, min_sums)
            if new_disc < current_disc:
                current_disc = new_disc
                no_improve = 0
                found_improvement = True
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                _revert_flip(signs, flip_set, max_sums, min_sums)
                no_improve += 1

    # --- Phase 4: Final Polish with Single Flips ---
    # Sometimes single flips are the last step
    if time.time() < deadline - 2:
        signs = best_signs[:]
        max_sums, min_sums = _compute_ap_sums(signs)
        current_disc = _get_disc(max_sums, min_sums)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(n):
                if time.time() > deadline - 2:
                    break
                flip_set = [idx]
                new_disc = _apply_flip(signs, flip_set, max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    _revert_flip(signs, flip_set, max_sums, min_sums)

    return best_signs
