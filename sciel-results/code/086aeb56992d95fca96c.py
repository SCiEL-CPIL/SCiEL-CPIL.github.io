"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: 150s limit. We stop searching at 145s to ensure return.
    deadline = start_time + 145.0

    N = 256

    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= N, d >= 1
    # Store 0-indexed positions
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            # 0-indexed
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)

    # For each position, list of AP indices that contain it
    # aps_at[i] = list of AP indices where position i appears
    aps_at = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            aps_at[pos].append(idx)

    # Convert to numpy arrays for speed
    # Also precompute the indices of the OTHER 3 positions for each AP relative to each position
    # This is complex to store fully, so we will compute on the fly using the aps array, 
    # but we will avoid copying the whole col array.
    
    aps_at_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in aps_at]

    def count_mono(col):
        """Count monochromatic 4-APs given color array col (0/1)."""
        c = col[aps]  # shape (num_aps, 4)
        # An AP is monochromatic if all 4 are 0 or all 4 are 1
        # Check if all elements in the row are equal
        # Using min/max is faster than multiple comparisons
        min_c = np.min(c, axis=1)
        max_c = np.max(c, axis=1)
        return int(np.sum(min_c == max_c))

    def fast_delta_single(col, i, new_val):
        """
        Compute change in mono count if we flip position i to new_val.
        Optimization: Do not copy col. Analyze affected APs.
        """
        affected = aps_at_arrays[i]
        if len(affected) == 0:
            return 0

        delta = 0
        # Get the APs containing i
        sub_aps = aps[affected] # shape (k, 4)
        
        # We need to check the status of these APs before and after flip.
        # Before flip:
        # An AP is mono if all 4 are same.
        # Since we are flipping i, the AP is mono BEFORE if:
        # 1. The other 3 positions have the same color as col[i].
        # After flip:
        # The AP is mono AFTER if:
        # 1. The other 3 positions have the same color as new_val (which is 1 - col[i]).
        
        # Let's extract the other 3 positions for each AP in sub_aps
        # We can do this by masking out the column where the value is i.
        # But finding the column index is annoying in numpy without loops.
        # However, k is small (~10-20). A python loop over the affected APs is fine and faster than numpy overhead for such small arrays? 
        # Actually, numpy vectorization on small arrays has overhead. Let's try a hybrid.
        
        # Let's stick to a very tight loop for the affected APs, which is small.
        # This avoids the col.copy() cost entirely.
        
        for ap_idx in affected:
            ap = aps[ap_idx]
            # Find the index of i in ap
            # ap is small (4 elements)
            # We can just check the other 3
            # Get the 3 positions that are not i
            others = []
            for p in ap:
                if p != i:
                    others.append(p)
            
            c1, c2, c3 = col[others[0]], col[others[1]], col[others[2]]
            
            # Check if other 3 are equal
            if c1 == c2 and c2 == c3:
                # The other 3 are all c1
                if col[i] == c1:
                    # Currently mono. Flip makes it non-mono. Delta -1
                    delta -= 1
                else:
                    # Currently non-mono (i is different). Flip makes it mono (i becomes c1). Delta +1
                    delta += 1
            # If other 3 are not all equal, delta is 0 (cannot become mono)
            
        return delta

    def fast_delta_pair(col, i, j):
        """
        Compute change in mono count if we flip positions i and j.
        """
        if i == j:
            return 0
            
        # Union of affected APs
        set_i = set(aps_at_arrays[i].tolist())
        set_j = set(aps_at_arrays[j].tolist())
        affected = list(set_i.union(set_j))
        
        if not affected:
            return 0
            
        delta = 0
        # We need to check each affected AP.
        # For an AP to be mono, all 4 must be same.
        # Case 1: AP contains i but not j. Flip i only.
        # Case 2: AP contains j but not i. Flip j only.
        # Case 3: AP contains both i and j. Flip both.
        
        for ap_idx in affected:
            ap = aps[ap_idx]
            has_i = (i in ap)
            has_j = (j in ap)
            
            # Determine the colors of the 4 positions after flip
            # It's easier to determine if it was mono before and after.
            
            # Before:
            # Check if all 4 are same
            c0, c1, c2, c3 = col[ap[0]], col[ap[1]], col[ap[2]], col[ap[3]]
            was_mono = (c0 == c1 == c2 == c3)
            
            # After:
            # Apply flips
            nc0 = c0
            nc1 = c1
            nc2 = c2
            nc3 = c3
            
            # Map ap positions to indices 0..3
            # We know i and j are in ap.
            # Let's just create a temp array for the 4 values
            vals = np.array([c0, c1, c2, c3])
            
            # Find indices of i and j in ap
            idx_i = -1
            idx_j = -1
            for k, p in enumerate(ap):
                if p == i:
                    idx_i = k
                elif p == j:
                    idx_j = k
                    
            if idx_i != -1:
                vals[idx_i] = 1 - vals[idx_i]
            if idx_j != -1:
                vals[idx_j] = 1 - vals[idx_j]
                
            is_mono = (vals[0] == vals[1] == vals[2] == vals[3])
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
                
        return delta

    def sa_run(seed, max_steps, start_temp, end_temp):
        """Run simulated annealing. Returns best color array and its score."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        # Start with random coloring
        col = np_rng.randint(0, 2, size=N).astype(np.int32)
        current_score = count_mono(col)
        best_col = col.copy()
        best_score = current_score

        steps_done = 0
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
                
            # Temperature schedule (geometric cooling)
            t = start_temp * (end_temp / start_temp) ** (step / max_steps)

            # Pick a random position to flip
            i = rng.randint(0, N - 1)
            new_val = 1 - col[i]

            # Compute delta
            delta = fast_delta_single(col, i, new_val)

            if delta <= 0:
                # Accept
                col[i] = new_val
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_col = col.copy()
            else:
                # Accept with probability exp(-delta / t)
                if t > 0 and rng.random() < np.exp(-delta / t):
                    col[i] = new_val
                    current_score += delta

        return best_col, best_score

    def local_search_single(col, current_score, max_passes=10):
        """Iterative improvement using single flips."""
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 10.0:
            improved = False
            passes += 1
            # Shuffle order to avoid bias
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 10.0:
                    break
                new_val = 1 - col[i]
                delta = fast_delta_single(col, i, new_val)
                if delta < 0:
                    col[i] = new_val
                    current_score += delta
                    improved = True
        return col, current_score

    def local_search_pair(col, current_score, max_passes=5):
        """Iterative improvement using pair flips. Slower but can escape local minima."""
        improved = True
        passes = 0
        # We can't check all pairs (256*256). We sample pairs.
        while improved and passes < max_passes and time.time() < deadline - 5.0:
            improved = False
            passes += 1
            # Try a subset of pairs
            # Random pairs
            for _ in range(500):
                if time.time() > deadline - 5.0:
                    break
                i, j = random.sample(range(N), 2)
                delta = fast_delta_pair(col, i, j)
                if delta < 0:
                    col[i] = 1 - col[i]
                    col[j] = 1 - col[j]
                    current_score += delta
                    improved = True
        return col, current_score

    # Multiple restarts
    best_overall = None
    best_overall_score = float('inf')

    # 1. Try structured colorings
    structured_patterns = [
        np.array([(i // 2) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i // 4) % 2 for i in range(N)], dtype=np.int32),
        np.array([(i // 8) % 2 for i in range(N)], dtype=np.int32),
        # Thue-Morse like?
        np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32),
    ]
    
    for pat in structured_patterns:
        s = count_mono(pat)
        if s < best_overall_score:
            best_overall_score = s
            best_overall = pat.copy()
            
    # 2. SA runs
    # We have ~145s. Precomputation is fast.
    # Let's allocate time for SA.
    # Each SA run with 100k steps should be fast now due to O(1) delta.
    # Let's do 5-10 restarts.
    
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 20.0: # Keep 20s for local search
            break

        seed = 100 + r * 123
        max_steps = 200000 # More steps possible now
        start_temp = 3.0
        end_temp = 0.001

        best_col, best_score = sa_run(seed, max_steps, start_temp, end_temp)

        # Polish with local search
        best_col, best_score = local_search_single(best_col, best_score, max_passes=20)
        best_col, best_score = local_search_pair(best_col, best_score, max_passes=3)
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall = best_col.copy()

    # 3. Final polish on the best overall
    if best_overall is not None and time.time() < deadline - 2.0:
        col = best_overall.copy()
        current_score = best_overall_score
        col, current_score = local_search_single(col, current_score, max_passes=50)
        col, current_score = local_search_pair(col, current_score, max_passes=10)
        
        if current_score < best_overall_score:
            best_overall_score = current_score
            best_overall = col

    return [int(c) for c in best_overall]
# EVOLVE-BLOCK-END
