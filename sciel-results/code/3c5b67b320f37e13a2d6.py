"""Rewritten program: simulated annealing with vectorized 4-AP penalty for 2-coloring [1,256]."""
import time
import numpy as np

def construct():
    n = 256
    # Deadline guard: aim to finish well before 120s
    deadline = time.time() + 115.0

    # Precompute all 4-term APs as index arrays (0-based)
    # AP: (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= n
    # In 0-based: i, i+d, i+2d, i+3d with 0 <= i and i+3d < n
    aps = []
    for d in range(1, n // 3 + 1):
        for i in range(n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    aps = np.array(aps, dtype=np.int32)
    num_aps = len(aps)

    # Helper: count monochromatic 4-APs for a given coloring (0/1 array, 0-based)
    def count_mono(color):
        # color is a numpy array of 0/1, length n
        # For each AP, check if all 4 positions have the same color
        c0 = color[aps[:, 0]]
        c1 = color[aps[:, 1]]
        c2 = color[aps[:, 2]]
        c3 = color[aps[:, 3]]
        # Monochromatic if all equal: (c0==c1) & (c1==c2) & (c2==c3)
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono))

    # Initialize with Thue-Morse sequence (known to minimize short APs)
    # Thue-Morse: t(i) = parity of number of 1s in binary representation of i
    # For position i (0-based), color = popcount(i) % 2
    color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    best_color = color.copy()
    best_score = count_mono(best_color)
    current_score = best_score

    # Simulated annealing parameters
    T_start = 10.0
    T_end = 0.01
    cooling = 0.995
    T = T_start

    # Vectorized delta computation for a single flip at position p
    # When we flip color[p], it affects all APs that include position p.
    # Precompute for each position, which APs it belongs to.
    # pos_to_aps[p] = list of AP indices where p appears
    pos_to_aps = [np.array([], dtype=np.int32) for _ in range(n)]
    for idx in range(num_aps):
        for k in range(4):
            p = aps[idx, k]
            pos_to_aps[p] = np.append(pos_to_aps[p], idx)

    # For faster delta computation, precompute the "contribution" of each AP
    # An AP is monochromatic if all 4 colors are equal.
    # When flipping position p in AP (i1,i2,i3,i4), the AP's mono status changes.

    rng = np.random.default_rng(42)

    while time.time() < deadline:
        # Pick a random position to flip
        p = rng.integers(0, n)
        
        # Compute delta in score if we flip position p
        # Get all APs containing p
        ap_indices = pos_to_aps[p]
        if len(ap_indices) == 0:
            continue
        
        # For each such AP, determine if it was mono before and after flip
        # Before: all 4 colors equal
        # After: color[p] is flipped, so check if the other 3 are equal to the new color[p]
        
        # Current colors of the 4 positions in each AP
        c0 = color[aps[ap_indices, 0]]
        c1 = color[aps[ap_indices, 1]]
        c2 = color[aps[ap_indices, 2]]
        c3 = color[aps[ap_indices, 3]]
        
        # Determine which position in the AP is p
        # For each AP, find which of the 4 slots corresponds to p
        # This is tricky to vectorize cleanly, so let's use a different approach
        
        # Alternative: just compute the score change directly
        # mono_before = all 4 equal
        mono_before = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # After flipping p, the new color at p is 1 - color[p]
        new_cp = 1 - color[p]
        
        # For each AP, the new colors depend on which slot p is in
        # Let's compute this by checking each slot
        # slot0_is_p = aps[ap_indices, 0] == p
        # etc.
        
        # Build new color arrays for the affected APs
        # Create a copy of the 4-tuple colors, then modify the position corresponding to p
        new_c0 = c0.copy()
        new_c1 = c1.copy()
        new_c2 = c2.copy()
        new_c3 = c3.copy()
        
        # For each AP, determine which slot has position p
        # We can do this with comparisons
        mask0 = (aps[ap_indices, 0] == p)
        mask1 = (aps[ap_indices, 1] == p)
        mask2 = (aps[ap_indices, 2] == p)
        mask3 = (aps[ap_indices, 3] == p)
        
        new_c0[mask0] = new_cp
        new_c1[mask1] = new_cp
        new_c2[mask2] = new_cp
        new_c3[mask3] = new_cp
        
        mono_after = (new_c0 == new_c1) & (new_c1 == new_c2) & (new_c2 == new_c3)
        
        delta = int(np.sum(mono_after)) - int(np.sum(mono_before))
        
        # Accept if delta <= 0 or with probability exp(-delta/T)
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            color[p] = 1 - color[p]
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_color = color.copy()
        
        # Cool down
        T *= cooling
        if T < T_end:
            T = T_end

    # Return best color as a list of 0/1 for positions 1..256
    # The task asks for colors of integers 1..256
    # Our array is 0-based, so color[0] corresponds to integer 1, etc.
    return [int(best_color[i]) for i in range(n)]
