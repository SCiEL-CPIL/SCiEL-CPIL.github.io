"""Direct heuristic construction for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: numpy array of +1/-1 indexed 0..n-1.
    Returns the max absolute sum.
    Uses vectorized NumPy for speed.
    """
    n = len(signs)
    max_abs = 0
    
    for d in range(1, n):
        # For each residue class r in 0..d-1, the subsequence is signs[r], signs[r+d], ...
        # We need max absolute subarray sum over each such subsequence.
        for r in range(d):
            if r >= n:
                continue
            # Get the subsequence as a numpy array
            idx = np.arange(r, n, d)
            sub = signs[idx]
            m = len(sub)
            if m <= 1:
                # Single element: discrepancy is just |sign| = 1
                if 1 > max_abs:
                    max_abs = 1
                continue
            # Compute cumulative sum
            cum = np.concatenate(([0], np.cumsum(sub)))
            # Max subarray sum = max(cum[j] - cum[i]) for j > i
            # = max over j of (cum[j] - min_{i<j} cum[i])
            # Vectorized: for each j, min prefix before j
            # Running min of cum[0..j-1] for j=1..m
            # Use a simple approach: compute running min
            # cum has length m+1
            # For position j (1-indexed in cum, i.e., cum[j]), the min prefix is min(cum[0..j-1])
            # We can compute this with a loop or use the fact that:
            # max_subarray = max(cum) - min(cum) is NOT correct in general
            # Correct: max_{0<=i<j<=m} (cum[j] - cum[i])
            
            # Efficient: track running min
            # cum_running_min[j] = min(cum[0..j-1]) for j=1..m
            # Then candidate_max = cum[1..m] - cum_running_min[1..m]
            # Similarly for min: cum_running_max[j] = max(cum[0..j-1])
            # candidate_min = cum[1..m] - cum_running_max[1..m]
            
            # Compute running min of cum[0..j-1] for each j
            # This is a prefix-min operation
            # cum_running_min[j] = min(cum[0], cum[1], ..., cum[j-1])
            # We can do this iteratively but vectorized approach:
            # Actually, let's just use the standard O(m) algorithm but with numpy for the subarray extraction
            
            # For small m (which is common when d is large), a simple Python loop might be faster
            # But let's try numpy
            # running_min[j] = min(cum[0:j]) for j in 1..m
            # We can compute this by: 
            #   cum_expanded = cum[np.newaxis, :]  # shape (1, m+1)
            #   But that's O(m^2) memory
            
            # Better: use the fact that we can compute it in a single pass
            # Let's use a simple loop for the running min/max since m is typically small
            # for large d, m is small; for small d, m is large but there are few such d
            
            if m <= 50:
                # Python loop is fine for small m
                min_pref = cum[0]
                max_pref = cum[0]
                max_sum = 0
                min_sum = 0
                for j in range(1, m + 1):
                    cj = cum[j]
                    if cj < min_pref:
                        min_pref = cj
                    if cj > max_pref:
                        max_pref = cj
                    diff_pos = cj - min_pref
                    diff_neg = cj - max_pref
                    if diff_pos > max_sum:
                        max_sum = diff_pos
                    if diff_neg < min_sum:
                        min_sum = diff_neg
                current_max = max(max_sum, -min_sum)
            else:
                # Use numpy for larger subsequences
                # Compute running min: min(cum[0:j]) for j=1..m
                # We can do this with a cumulative minimum trick
                # Actually, numpy doesn't have a direct "running min excluding current" 
                # Let's use: for each j, we want min(cum[0:j])
                # This is equivalent to: shift cum by 1 and take cumulative min
                # cum_shifted[j] = min(cum[0], cum[1], ..., cum[j-1])
                # We can compute: cum_min_prefix = np.minimum.accumulate(cum) gives min(cum[0..j])
                # But we need min(cum[0..j-1]) for position j
                # So: cum_min_prefix[j-1] = min(cum[0..j-1])
                # Let cum_min_acc = np.minimum.accumulate(cum)  # cum_min_acc[j] = min(cum[0..j])
                # Then for j>=1, min(cum[0..j-1]) = cum_min_acc[j-1]
                
                cum_min_acc = np.minimum.accumulate(cum)  # length m+1
                cum_max_acc = np.maximum.accumulate(cum)
                
                # For j=1..m: min_before_j = cum_min_acc[j-1], max_before_j = cum_max_acc[j-1]
                min_before = cum_min_acc[:m]  # cum_min_acc[0..m-1], length m
                max_before = cum_max_acc[:m]
                
                cum_tail = cum[1:]  # cum[1..m], length m
                
                max_sums = cum_tail - min_before  # max subarray sum ending at each position
                min_sums = cum_tail - max_before  # min subarray sum ending at each position
                
                max_sum = np.max(max_sums)
                min_sum = np.min(min_sums)
                current_max = max(max_sum, -min_sum)
            
            if current_max > max_abs:
                max_abs = current_max
                # Early exit: if max_abs is already high, we could stop
                # But for search purposes, we need the exact value
                # However, if we're just checking if a flip improves things,
                # we can early-exit if max_abs exceeds current best
                # For now, compute fully
    
    return int(max_abs)

def _max_ap_discrepancy_fast(signs, early_exit_val=None):
    """Faster version that can early-exit if discrepancy exceeds early_exit_val."""
    n = len(signs)
    max_abs = 0
    
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            idx = np.arange(r, n, d)
            sub = signs[idx]
            m = len(sub)
            if m <= 1:
                if 1 > max_abs:
                    max_abs = 1
                continue
            
            cum = np.concatenate(([0], np.cumsum(sub)))
            
            if m <= 50:
                min_pref = cum[0]
                max_pref = cum[0]
                max_sum = 0
                min_sum = 0
                for j in range(1, m + 1):
                    cj = cum[j]
                    if cj < min_pref:
                        min_pref = cj
                    if cj > max_pref:
                        max_pref = cj
                    diff_pos = cj - min_pref
                    diff_neg = cj - max_pref
                    if diff_pos > max_sum:
                        max_sum = diff_pos
                    if diff_neg < min_sum:
                        min_sum = diff_neg
                current_max = max(max_sum, -min_sum)
            else:
                cum_min_acc = np.minimum.accumulate(cum)
                cum_max_acc = np.maximum.accumulate(cum)
                min_before = cum_min_acc[:m]
                max_before = cum_max_acc[:m]
                cum_tail = cum[1:]
                max_sums = cum_tail - min_before
                min_sums = cum_tail - max_before
                max_sum = np.max(max_sums)
                min_sum = np.min(min_sums)
                current_max = max(max_sum, -min_sum)
            
            if current_max > max_abs:
                max_abs = current_max
                if early_exit_val is not None and max_abs >= early_exit_val:
                    return max_abs
    
    return int(max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    rng = random.Random(42)
    
    # Generate several initial candidates
    candidates = []
    
    # Random
    signs_rand = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
    candidates.append(signs_rand)
    
    # Thue-Morse
    signs_tm = np.zeros(n, dtype=np.int8)
    for i in range(n):
        t = 0
        x = i
        while x:
            t ^= (x & 1)
            x >>= 1
        signs_tm[i] = 1 if t == 0 else -1
    candidates.append(signs_tm)
    
    # Alternating
    signs_alt = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    candidates.append(signs_alt)
    
    # More random starts
    for seed in [123, 456, 789, 101, 202, 303]:
        r2 = random.Random(seed)
        candidates.append(np.array([r2.choice((-1, 1)) for _ in range(n)], dtype=np.int8))
    
    # Evaluate all candidates and pick the best
    best_signs = None
    best_disc = float('inf')
    for c in candidates:
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c.copy()
    
    signs = best_signs.copy()
    current_disc = best_disc
    
    # Hill climbing with simulated annealing flavor
    # Accept improvements always; accept worsenings with small probability early on
    max_rounds = 5000
    round_count = 0
    no_improve_streak = 0
    max_no_improve = 100  # After this many rounds without improvement, tighten acceptance
    
    while time.time() < deadline and round_count < max_rounds:
        round_count += 1
        improved_this_round = False
        
        # Try flipping each position in random order
        order = np.random.default_rng(round_count * 42 + 1).permutation(n)
        
        for i in order:
            if time.time() > deadline:
                break
            
            old_val = signs[i]
            signs[i] = -old_val
            
            # Use fast evaluator with early exit at current_disc
            # If new disc >= current_disc, we can early exit
            new_disc = _max_ap_discrepancy_fast(signs, early_exit_val=current_disc)
            
            if new_disc < current_disc:
                current_disc = new_disc
                best_signs = signs.copy()
                best_disc = current_disc
                improved_this_round = True
                no_improve_streak = 0
            else:
                # Consider accepting with small probability if close
                # Only if we haven't improved for a while (annealing)
                if no_improve_streak > 20 and new_disc == current_disc:
                    # Same discrepancy, accept with small prob to explore
                    if rng.random() < 0.1:
                        improved_this_round = True  # Count as exploration
                    else:
                        signs[i] = old_val
                else:
                    signs[i] = old_val
        
        if improved_this_round:
            no_improve_streak = 0
        else:
            no_improve_streak += 1
        
        # If we've been stuck for a long time, try a random restart from best
        if no_improve_streak > 50 and round_count % 10 == 0:
            # Perturb the best solution
            signs = best_signs.copy()
            # Flip a few random positions
            num_flips = rng.randint(3, 10)
            for _ in range(num_flips):
                pos = rng.randint(0, n - 1)
                signs[pos] = -signs[pos]
            current_disc = _max_ap_discrepancy(signs)
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
            no_improve_streak = 0
    
    # Final polish: strict hill climbing from best solution
    signs = best_signs.copy()
    current_disc = best_disc
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            old_val = signs[i]
            signs[i] = -old_val
            new_disc = _max_ap_discrepancy_fast(signs, early_exit_val=current_disc)
            
            if new_disc < current_disc:
                current_disc = new_disc
                best_disc = current_disc
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = old_val
        
        if not improved:
            break
    
    return best_signs.tolist()
# EVOLVE-BLOCK-END
