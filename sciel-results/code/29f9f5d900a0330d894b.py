import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all arithmetic progressions of length >= 2
    # We'll store them as arrays of indices (0-based)
    aps = []
    for d in range(1, N // 2 + 1):
        for start in range(0, N - d):  # 0-based start
            # AP: start, start+d, start+2d, ...
            indices = list(range(start, N, d))
            if len(indices) >= 2:
                aps.append(indices)
    
    # Also note: singletons have sum ±1, so max_abs_sum >= 1 always.
    # We're minimizing the max over all APs including singletons, so the lower bound is 1.
    
    # Convert to numpy array for fast dot products
    # Each AP is a row of 0/1 matrix
    # But there could be many APs. Let's estimate:
    # For d=1: N-1 = 255 APs
    # For d=2: ~127 APs
    # Total is roughly N log N / 2 or so
    
    # Let's use a different approach: for each position, track which APs contain it
    # Actually, let's just build a sparse representation
    
    # Alternative: use numpy arrays and compute sums efficiently
    # Number of APs: sum over d=1 to N/2 of (N-d) roughly = N^2/4 - N/2 ≈ 16384 - 128 ≈ 16256
    # That's manageable with numpy if we're careful
    
    # Let's build a list of AP index arrays
    ap_arrays = [np.array(a) for a in aps]
    
    def compute_max_abs_sum(signs):
        """Compute max absolute sum over all APs."""
        max_abs = 1  # Singletons guarantee at least 1
        for ap in ap_arrays:
            s = np.sum(signs[ap])
            abs_s = abs(s)
            if abs_s > max_abs:
                max_abs = abs_s
        return max_abs
    
    # Start with a random assignment
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(N)])
    
    best_signs = signs.copy()
    best_max = compute_max_abs_sum(signs)
    
    # Local search: flip individual bits and see if it improves
    def try_flip(signs, i, current_max):
        """Try flipping bit i, return (new_max, improved)."""
        signs[i] = -signs[i]
        new_max = compute_max_abs_sum(signs)
        signs[i] = -signs[i]  # revert
        return new_max, new_max < current_max
    
    # Simulated annealing or simple hill climbing
    # Given time constraints, let's do a simple local search with multiple restarts
    
    # Precompute for each position, which APs it belongs to, to speed up evaluation
    # Actually, let's just do a simple approach: iterate through positions and try flipping
    
    # Better approach: use the fact that flipping one sign changes the sum of all APs containing that position
    # Let's precompute: for each position i, list of AP indices that contain i
    
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(ap_idx)
    
    # Now, for each AP, we can compute its sum quickly
    # But we still need to check all APs when we flip
    
    # Let's try a different strategy: use the precomputed pos_to_aps to update sums incrementally
    
    # Initialize AP sums
    ap_sums = np.zeros(len(aps), dtype=int)
    for ap_idx, ap in enumerate(aps):
        ap_sums[ap_idx] = np.sum(signs[ap])
    
    current_max = int(np.max(np.abs(ap_sums)))
    
    def compute_max_from_sums(ap_sums):
        return int(np.max(np.abs(ap_sums)))
    
    # Local search with incremental updates
    def flip_and_update(signs, i, ap_sums):
        """Flip sign at position i and update ap_sums incrementally."""
        old_sign = signs[i]
        signs[i] = -signs[i]
        delta = -2 * old_sign  # Change in sum for each AP containing i
        for ap_idx in pos_to_aps[i]:
            ap_sums[ap_idx] += delta
        return signs[i]
    
    # Simple hill climbing
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() - start_time > deadline:
                break
            old_sign = signs[i]
            flip_and_update(signs, i, ap_sums)
            new_max = compute_max_from_sums(ap_sums)
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
                improved = True
            else:
                # Revert
                flip_and_update(signs, i, ap_sums)
        if time.time() - start_time > deadline:
            break
    
    # Try a few more random restarts if time permits
    while time.time() - start_time < deadline - 5:
        # Random restart
        signs = np.array([rng.choice([-1, 1]) for _ in range(N)])
        ap_sums = np.zeros(len(aps), dtype=int)
        for ap_idx, ap in enumerate(aps):
            ap_sums[ap_idx] = np.sum(signs[ap])
        
        current_max = compute_max_from_sums(ap_sums)
        
        # Quick local search
        for _ in range(100):  # Max 100 iterations per restart
            if time.time() - start_time > deadline:
                break
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            flip_and_update(signs, i, ap_sums)
            new_max = compute_max_from_sums(ap_sums)
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
            else:
                flip_and_update(signs, i, ap_sums)
    
    return list(best_signs)
