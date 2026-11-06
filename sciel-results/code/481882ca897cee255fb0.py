import numpy as np
import random
import time
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

    # Efficient precomputation: build position -> list of AP indices containing it
    # This is O(num_aps) instead of O(N * num_aps)
    pos_to_aps = [[] for _ in range(N)]
    for idx in range(num_aps):
        for j in range(4):
            pos = aps[idx, j]
            pos_to_aps[pos].append(idx)

    # For each position, precompute the AP indices as a numpy array
    others_for_pos = [None] * N
    for i in range(N):
        ap_indices = pos_to_aps[i]
        if ap_indices:
            others_for_pos[i] = np.array(ap_indices, dtype=np.int32)
        else:
            others_for_pos[i] = np.array([], dtype=np.int32)

    def count_mono(col):
        """Count monochromatic 4-APs given color array col (0/1)."""
        c = col[aps]  # shape (num_aps, 4)
        # Check if all 4 are equal
        # min == max is a fast way
        return int(np.sum(np.min(c, axis=1) == np.max(c, axis=1)))

    def fast_delta_single(col, i):
        """
        Compute change in mono count if we flip position i.
        """
        ap_indices = others_for_pos[i]
        if len(ap_indices) == 0:
            return 0
        
        # Get the 4 positions for each affected AP
        affected_aps = aps[ap_indices]  # shape (k, 4)
        
        # Get colors of all 4 positions
        all_colors = col[affected_aps]  # shape (k, 4)
        
        # Sum of colors at non-i positions
        total_sum = np.sum(all_colors, axis=1)  # shape (k,)
        c_i = col[i]
        others_sum = total_sum - c_i  # sum of the other 3 colors
        
        # The 3 others are all 0 if others_sum == 0, all 1 if others_sum == 3
        all_zero = (others_sum == 0)
        all_one = (others_sum == 3)
        all_same = all_zero | all_one
        
        if not np.any(all_same):
            return 0
            
        # others_color is 0 where all_zero, 1 where all_one
        others_color = all_one.astype(np.int32)
        
        mono_before = all_same & (c_i == others_color)
        mono_after = all_same & ((1 - c_i) == others_color)
        
        count_before = np.sum(mono_before)
        count_after = np.sum(mono_after)
        
        return int(count_after - count_before)

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
            if time.time() > deadline - 2.0:
                break
                
            # Temperature schedule (geometric cooling)
            t = start_temp * (end_temp / start_temp) ** (step / max_steps)

            # Pick a random position to flip
            i = rng.randint(0, N - 1)
            
            # Compute delta
            delta = fast_delta_single(col, i)

            if delta <= 0:
                # Accept
                col[i] = 1 - col[i]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_col = col.copy()
            else:
                # Accept with probability exp(-delta / t)
                if t > 0 and rng.random() < np.exp(-delta / t):
                    col[i] = 1 - col[i]
                    current_score += delta

        return best_col, best_score

    def local_search_single(col, current_score, max_passes=10):
        """Iterative improvement using single flips."""
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 5.0:
            improved = False
            passes += 1
            # Shuffle order to avoid bias
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                delta = fast_delta_single(col, i)
                if delta < 0:
                    col[i] = 1 - col[i]
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
        np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32),
    ]
    
    for pat in structured_patterns:
        s = count_mono(pat)
        if s < best_overall_score:
            best_overall_score = s
            best_overall = pat.copy()
            
    # 2. SA runs
    # With faster precomputation, we have more time for search
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        seed = 100 + r * 123
        max_steps = 80000
        start_temp = 3.0
        end_temp = 0.01

        best_col, best_score = sa_run(seed, max_steps, start_temp, end_temp)

        # Polish with local search
        best_col, best_score = local_search_single(best_col, best_score, max_passes=15)
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall = best_col.copy()

    # 3. Final polish on the best overall
    if best_overall is not None and time.time() < deadline - 1.0:
        col = best_overall.copy()
        current_score = best_overall_score
        col, current_score = local_search_single(col, current_score, max_passes=25)
        
        if current_score < best_overall_score:
            best_overall_score = current_score
            best_overall = col

    return [int(c) for c in best_overall]
# EVOLVE-BLOCK-END
