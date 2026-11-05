import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    n = 256

    # Build all APs with length >= 1
    # An AP is defined by start r (0 <= r < d) and step d (1 <= d < n)
    # The sequence is indices r, r+d, r+2d, ... < n
    aps = []
    ap_membership = [[] for _ in range(n)]
    
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 1:
                ap_idx = len(aps)
                aps.append(sub_indices)
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)
    
    num_aps = len(aps)
    
    def _kade_max_min(vals):
        """Compute max and min subarray sum for a list of values."""
        if len(vals) == 0:
            return 0, 0
        max_ending = 0
        max_so_far = vals[0]
        min_ending = 0
        min_so_far = vals[0]
        
        for v in vals:
            max_ending = max(0, max_ending + v)
            max_so_far = max(max_so_far, max_ending)
            
            min_ending = min(0, min_ending + v)
            min_so_far = min(min_so_far, min_ending)
            
        return max_so_far, min_so_far

    def _compute_ap_stats(signs):
        """Compute max/min subarray sums for all APs."""
        max_sums = np.zeros(num_aps, dtype=np.int32)
        min_sums = np.zeros(num_aps, dtype=np.int32)
        for i in range(num_aps):
            ap = aps[i]
            vals = [signs[idx] for idx in ap]
            m, mn = _kade_max_min(vals)
            max_sums[i] = m
            min_sums[i] = mn
        return max_sums, min_sums

    def _get_disc(max_sums, min_sums):
        if num_aps == 0:
            return 0
        return max(int(np.max(max_sums)), -int(np.min(min_sums)), 1)

    def _update_ap(ap_idx, signs):
        """Update stats for a single AP."""
        ap = aps[ap_idx]
        vals = [signs[idx] for idx in ap]
        m, mn = _kade_max_min(vals)
        return m, mn

    # --- Phase 1: Generate diverse candidates ---
    candidates = []

    # Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)

    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(alt)

    # Random
    for s in range(10):
        r = random.Random(s)
        candidates.append([r.choice((-1, 1)) for _ in range(n)])

    # --- Phase 2: Evaluate and pick best start ---
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    for cand in candidates:
        if time.time() > deadline - 20:
            break
        max_sums, min_sums = _compute_ap_stats(cand)
        d = _get_disc(max_sums, min_sums)
        if d < best_disc:
            best_disc = d
            best_signs = cand[:]

    # --- Phase 3: Iterated Simulated Annealing ---
    rng = random.Random(42)
    
    # We will run multiple independent SA runs
    # Each run starts from the current best or a perturbed version
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize state for this restart
        if restart == 0:
            signs = best_signs[:]
        else:
            # Perturb the best solution found so far
            signs = best_signs[:]
            num_flips = n // 10  # 10% perturbation
            for _ in range(num_flips):
                idx = rng.randint(0, n - 1)
                signs[idx] = -signs[idx]
                
        max_sums, min_sums = _compute_ap_stats(signs)
        current_disc = _get_disc(max_sums, min_sums)
        
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs[:]

        # SA Parameters
        temp = 2.0
        cool = 0.99
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            # Pick a random index to flip
            idx = rng.randint(0, n - 1)
            old_val = signs[idx]
            signs[idx] = -old_val
            
            # Update affected APs
            for ap_idx in ap_membership[idx]:
                new_m, new_mn = _update_ap(ap_idx, signs)
                max_sums[ap_idx] = new_m
                min_sums[ap_idx] = new_mn
                
            new_disc = _get_disc(max_sums, min_sums)
            
            if new_disc < current_disc:
                current_disc = new_disc
                no_improve = 0
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                delta = new_disc - current_disc
                if delta > 0 and temp > 0.05:
                    if rng.random() < math.exp(-delta / temp):
                        current_disc = new_disc
                        no_improve = 1
                    else:
                        # Revert
                        signs[idx] = old_val
                        for ap_idx in ap_membership[idx]:
                            new_m, new_mn = _update_ap(ap_idx, signs)
                            max_sums[ap_idx] = new_m
                            min_sums[ap_idx] = new_mn
                        no_improve += 1
                else:
                    # Revert
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        new_m, new_mn = _update_ap(ap_idx, signs)
                        max_sums[ap_idx] = new_m
                        min_sums[ap_idx] = new_mn
                    no_improve += 1
                    
            temp *= cool

    # --- Phase 4: Final Polish ---
    if time.time() < deadline - 2:
        signs = best_signs[:]
        max_sums, min_sums = _compute_ap_stats(signs)
        current_disc = _get_disc(max_sums, min_sums)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try flipping each index
            for idx in range(n):
                old_val = signs[idx]
                signs[idx] = -old_val
                for ap_idx in ap_membership[idx]:
                    new_m, new_mn = _update_ap(ap_idx, signs)
                    max_sums[ap_idx] = new_m
                    min_sums[ap_idx] = new_mn
                new_disc = _get_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        new_m, new_mn = _update_ap(ap_idx, signs)
                        max_sums[ap_idx] = new_m
                        min_sums[ap_idx] = new_mn

    return best_signs
