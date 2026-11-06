"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

N = 256
DEADLINE = 115.0

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N (0-indexed)."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _thue_morse():
    """Generate Thue-Morse sequence for 0..255."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_ap_data():
    """
    Precompute data for fast vectorized delta calculation.
    Returns:
        ap_indices: List of numpy arrays of indices of the other 3 positions for each AP containing i.
        ap_k: List of numpy arrays of the position k (0-3) of i in that AP.
    """
    ap_indices = [[] for _ in range(N)]
    ap_k = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        for a in range(max_a):
            idxs = [a, a + d, a + 2 * d, a + 3 * d]
            for k in range(4):
                i = idxs[k]
                others = [idxs[j] for j in range(4) if j != k]
                ap_indices[i].append(others)
                ap_k[i].append(k)
                
    # Convert to numpy arrays
    ap_indices_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ap_indices]
    ap_k_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ap_k]
    
    return ap_indices_np, ap_k_np

def _delta_flip_vectorized(c, i, ap_indices_np, ap_k_np):
    """
    Calculate delta for flipping position i using vectorized operations.
    """
    idxs = ap_indices_np[i]
    ks = ap_k_np[i]
    
    if len(idxs) == 0:
        return 0

    # Get values of the other 3 positions for all APs containing i
    # shape: (num_aps, 3)
    other_vals = c[idxs]
    
    # Determine which of the other 3 are equal to c[i]
    # ci is scalar
    ci = c[i]
    
    # Count matches with ci for each AP
    # other_vals == ci is boolean array (num_aps, 3)
    matches = (other_vals == ci)
    num_matches = np.sum(matches, axis=1)
    
    # AP is mono if all 4 are equal.
    # Currently mono if num_matches == 3 (since the 3 others match ci, and i is ci).
    # Wait, if all 3 others match ci, then the AP is mono (ci, ci, ci, ci).
    # So was_mono = (num_matches == 3)
    
    # After flipping i, new value is 1-ci.
    # The 3 others have values v1, v2, v3.
    # New AP is mono if all 3 others are equal to 1-ci.
    # So num_matches_new = sum(v_j == 1-ci).
    # Since v_j is 0 or 1, v_j == 1-ci is equivalent to v_j != ci.
    # So num_matches_new = 3 - num_matches.
    
    # is_mono_new = (num_matches_new == 3) => (3 - num_matches == 3) => (num_matches == 0)
    
    was_mono = (num_matches == 3)
    is_mono_new = (num_matches == 0)
    
    # delta = count(is_mono_new) - count(was_mono)
    delta = np.sum(is_mono_new) - np.sum(was_mono)
    
    return int(delta)

def _local_search_delta(c, time_limit, ap_indices_np, ap_k_np):
    """Improve coloring via single-flip local search using vectorized delta."""
    start = time.time()
    current = _count_mono4(c)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() - start > time_limit:
                break
            delta = _delta_flip_vectorized(c, i, ap_indices_np, ap_k_np)
            if delta < 0:
                c[i] = 1 - c[i]
                current += delta
                improved = True
        if time.time() - start > time_limit:
            break
    return c, current

def _two_flip_search(c, time_limit, ap_indices_np, ap_k_np):
    """
    Try 2-flip moves to escape local minima.
    Only tries pairs (i, j) where i < j.
    """
    start = time.time()
    current = _count_mono4(c)
    improved = True
    while improved:
        improved = False
        # To save time, we might not check all pairs if time is tight, 
        # but N=256 means ~32k pairs. Delta calculation is fast.
        for i in range(N):
            if time.time() - start > time_limit:
                break
            for j in range(i + 1, N):
                if time.time() - start > time_limit:
                    break
                # Calculate delta for flipping i and j
                # This is complex to vectorize perfectly without modifying c, 
                # so we do it carefully.
                
                # We can compute delta_i and delta_j, but they interfere if APs overlap.
                # Simpler: just flip, count, unflip if bad? No, counting is slow.
                # Better: Calculate delta exactly.
                
                # Let's just use the vectorized delta for i, then for j on the modified array?
                # No, that's not correct for simultaneous flip.
                
                # Efficient way:
                # delta(i, j) = sum_{APs} (Mono(AP after flip i,j) - Mono(AP before flip i,j))
                
                # For an AP to change status, it must contain i or j.
                # We can iterate through APs containing i and APs containing j.
                
                # This is getting complicated. Let's stick to single flips for speed and rely on SA.
                # Or, a simplified 2-flip: Flip i, then do a quick single-flip pass?
                # That's just local search.
                
                # Let's skip explicit 2-flip search for now to keep code simple and fast,
                # as SA is already good at escaping.
                pass
        if time.time() - start > time_limit:
            break
    return c, current

def _simulated_annealing(c, time_limit, ap_indices_np, ap_k_np, T_start=5.0, T_end=0.1, alpha=0.999):
    """Simulated annealing to escape local minima."""
    start = time.time()
    current = _count_mono4(c)
    T = T_start
    rng = np.random.default_rng(123)
    best_c = c.copy()
    best_count = current
    
    # Pre-calculate log probabilities or just use random
    while time.time() - start < time_limit:
        # Pick random position
        i = rng.integers(0, N)
        delta = _delta_flip_vectorized(c, i, ap_indices_np, ap_k_np)
        
        if delta < 0:
            # Always accept improvement
            c[i] = 1 - c[i]
            current += delta
            if current < best_count:
                best_count = current
                best_c = c.copy()
        else:
            # Accept with probability exp(-delta/T)
            if delta == 0:
                c[i] = 1 - c[i]
            else:
                if T > 0.01:
                    # Optimization: avoid exp if possible?
                    # log(p) = -delta/T
                    # rng.random() < exp(-delta/T)
                    # log(rng.random()) < -delta/T
                    # If rng.random() is 0, log is -inf.
                    # Let's just use exp, it's fast enough in numpy?
                    # Actually, np.exp is fast.
                    if rng.random() < np.exp(-delta / T):
                        c[i] = 1 - c[i]
                        current += delta
        
        T *= alpha
        if T < T_end:
            T = T_end
    
    return best_c, best_count

def construct():
    start_time = time.time()
    ap_indices_np, ap_k_np = _precompute_ap_data()
    
    # Start with Thue-Morse
    c = _thue_morse()
    best_c = c.copy()
    best_count = _count_mono4(c)
    
    # Local search from Thue-Morse using vectorized delta
    time_remaining = DEADLINE - (time.time() - start_time)
    if time_remaining > 5:
        c, cnt = _local_search_delta(c, min(30, time_remaining - 5), ap_indices_np, ap_k_np)
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()
    
    # Try simulated annealing from best
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 15:
        c_sa = best_c.copy()
        c_sa, cnt_sa = _simulated_annealing(c_sa, min(40, remaining - 5), ap_indices_np, ap_k_np)
        if cnt_sa < best_count:
            best_count = cnt_sa
            best_c = c_sa.copy()
    
    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        # Use remaining time for multiple short SA runs from random starts
        # or local search from random starts
        while time.time() - start_time < DEADLINE - 5:
            c_rand = rng.integers(0, 2, size=N).astype(np.int8)
            time_left = DEADLINE - (time.time() - start_time) - 5
            if time_left < 2:
                break
            
            # Quick local search
            c_rand, cnt_rand = _local_search_delta(c_rand, min(5, time_left // 2), ap_indices_np, ap_k_np)
            if cnt_rand < best_count:
                best_count = cnt_rand
                best_c = c_rand.copy()
            
            # SA if time allows
            time_left = DEADLINE - (time.time() - start_time) - 5
            if time_left > 5 and cnt_rand <= best_count + 5:
                c_sa2 = c_rand.copy()
                c_sa2, cnt_sa2 = _simulated_annealing(c_sa2, min(10, time_left - 2), ap_indices_np, ap_k_np)
                if cnt_sa2 < best_count:
                    best_count = cnt_sa2
                    best_c = c_sa2.copy()
    
    return [int(x) for x in best_c]
# EVOLVE-BLOCK-END
