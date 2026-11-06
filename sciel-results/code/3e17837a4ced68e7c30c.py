import numpy as np
import random
import time
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: 120s limit. We stop searching at 115s to ensure return.
    deadline = start_time + 115.0

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

    # Precompute for each position i, the "other" positions in each AP containing i.
    # others_for_pos[i] will be a list of arrays, each of length 3.
    # Or better, a single numpy array of shape (num_aps_containing_i, 3) for each i.
    # We also need to know which AP index corresponds to which row, but for delta calculation
    # we only need the colors of the other 3 positions.
    
    others_for_pos = [None] * N
    for i in range(N):
        # Find all APs containing i
        # This is a bit slow to do naively, but N is small.
        # We can use the previously computed aps_at logic but store the others.
        others_list = []
        for idx in range(num_aps):
            if i in aps[idx]:
                others = [p for p in aps[idx] if p != i]
                others_list.append(others)
        if others_list:
            others_for_pos[i] = np.array(others_list, dtype=np.int32)
        else:
            others_for_pos[i] = np.array([], dtype=np.int32).reshape(0, 3)

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
        others = others_for_pos[i]
        if others.shape[0] == 0:
            return 0
        
        # Get colors of the other 3 positions for each affected AP
        # others shape: (k, 3)
        # col[others] shape: (k, 3)
        others_colors = col[others]
        
        # Check if the 3 others are all the same color
        # others_colors[:, 0] == others_colors[:, 1] == others_colors[:, 2]
        all_same = (others_colors[:, 0] == others_colors[:, 1]) & (others_colors[:, 1] == others_colors[:, 2])
        
        if not np.any(all_same):
            return 0
            
        # For those where others are same, let c_other = others_colors[:, 0]
        c_other = others_colors[:, 0]
        c_i = col[i]
        
        # Before flip: mono if c_other == c_i
        mono_before = c_other == c_i
        # After flip: mono if c_other == (1 - c_i)
        mono_after = c_other == (1 - c_i)
        
        # Apply mask all_same
        count_before = np.sum(mono_before & all_same)
        count_after = np.sum(mono_after & all_same)
        
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
    # Allocate time carefully. 
    # We have ~115s. Precomputation is done.
    # Let's do fewer restarts but longer runs, or more restarts with shorter runs.
    # The delta calculation is now vectorized and fast.
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10.0: # Keep 10s for final polish
            break

        seed = 100 + r * 123
        max_steps = 50000 # Reduced steps per run to allow more restarts or just finish faster
        start_temp = 2.0
        end_temp = 0.01

        best_col, best_score = sa_run(seed, max_steps, start_temp, end_temp)

        # Polish with local search
        best_col, best_score = local_search_single(best_col, best_score, max_passes=10)
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall = best_col.copy()

    # 3. Final polish on the best overall
    if best_overall is not None and time.time() < deadline - 1.0:
        col = best_overall.copy()
        current_score = best_overall_score
        col, current_score = local_search_single(col, current_score, max_passes=20)
        
        if current_score < best_overall_score:
            best_overall_score = current_score
            best_overall = col

    return [int(c) for c in best_overall]
# EVOLVE-BLOCK-END
