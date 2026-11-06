import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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

    # Precompute "others" for each position
    # others[i] is an array of shape (len(aps_at[i]), 3)
    # containing the indices of the other 3 positions in each AP containing i
    others = [None] * N
    for i in range(N):
        if not aps_at[i]:
            others[i] = np.array([], dtype=np.int32).reshape(0, 3)
            continue
        ap_indices = aps_at[i]
        sub_aps = aps[ap_indices]  # shape (M, 4)
        # We need to remove position i from each row
        # Find column index of i in each row
        # Since i is in the AP, we can find which column it is
        # cols = np.where(sub_aps == i)[1]
        # But it's easier to just mask
        # Create a mask for positions not equal to i
        mask = sub_aps != i
        # Extract the other 3 positions
        # This creates a (M, 4) array with 0s where i was, then we flatten and reshape?
        # Simpler:
        others_list = []
        for k in range(len(ap_indices)):
            row = sub_aps[k]
            # row contains i exactly once
            others_row = [p for p in row if p != i]
            others_list.append(others_row)
        others[i] = np.array(others_list, dtype=np.int32)

    def count_mono(col):
        """Count monochromatic 4-APs given color array col (0/1)."""
        c = col[aps]  # shape (num_aps, 4)
        all_zero = np.all(c == 0, axis=1)
        all_one = np.all(c == 1, axis=1)
        return int(np.sum(all_zero) + np.sum(all_one))

    def fast_delta(col, i, new_val):
        """Compute change in mono count if we flip position i to new_val."""
        others_i = others[i]
        if others_i.size == 0:
            return 0
        
        current_val = col[i]
        if current_val == new_val:
            return 0
            
        # Get colors of the other 3 positions in each affected AP
        # others_i has shape (M, 3)
        cols_others = col[others_i]  # shape (M, 3)
        
        # Count APs that become monochromatic (all new_val)
        # This happens if all 3 others are new_val
        all_new = np.all(cols_others == new_val, axis=1).sum()
        
        # Count APs that cease to be monochromatic (were all current_val)
        # This happens if all 3 others are current_val
        all_old = np.all(cols_others == current_val, axis=1).sum()
        
        return int(all_new - all_old)

    def sa_run(seed, max_steps, start_temp, end_temp):
        """Run simulated annealing. Returns best color array and its score."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        # Start with random coloring
        col = np_rng.randint(0, 2, size=N).astype(np.int32)
        current_score = count_mono(col)
        best_col = col.copy()
        best_score = current_score

        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break

            # Temperature schedule (geometric cooling)
            t = start_temp * (end_temp / start_temp) ** (step / max_steps)

            # Pick a random position to flip
            i = rng.randint(0, N - 1)
            new_val = 1 - col[i]

            # Compute delta
            delta = fast_delta(col, i, new_val)

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

    def local_search(col, max_iters=100):
        """Greedy local search: flip any position if it improves the score."""
        current_score = count_mono(col)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 1.0:
            improved = False
            iters += 1
            for i in range(N):
                if time.time() > deadline - 1.0:
                    break
                new_val = 1 - col[i]
                delta = fast_delta(col, i, new_val)
                if delta < 0:
                    col[i] = new_val
                    current_score += delta
                    improved = True
        return col, current_score

    # Multiple restarts
    best_overall = None
    best_overall_score = float('inf')

    # Try a few simple structured colorings first
    # 1. Random
    # 2. Alternating blocks
    # 3. Thue-Morse like (parity of popcount)
    
    # Thue-Morse
    tm = np.zeros(N, dtype=np.int32)
    for i in range(N):
        # Count bits set in i
        tm[i] = bin(i).count('1') % 2
    s = count_mono(tm)
    if s < best_overall_score:
        best_overall_score = s
        best_overall = tm.copy()
        
    # Alternating 2
    alt2 = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int32)
    s = count_mono(alt2)
    if s < best_overall_score:
        best_overall_score = s
        best_overall = alt2.copy()

    # Run SA with multiple seeds
    # We have ~145 seconds. 
    # Each SA run should be kept short to allow many restarts.
    # Let's estimate time per step.
    # fast_delta is vectorized over ~50 elements. Should be < 10us.
    # count_mono is ~13000 elements. ~50us.
    # SA loop overhead ~10us.
    # ~100us per step. 10000 steps = 1s.
    # So we can do ~100 restarts of 10000 steps? 
    # Let's try 50 restarts of 20000 steps.
    
    num_restarts = 60
    max_steps = 20000
    start_temp = 5.0
    end_temp = 0.01

    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        seed = 42 + r * 123
        best_col, best_score = sa_run(seed, max_steps, start_temp, end_temp)

        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall = best_col.copy()

    # Final local search polish on the best result found
    if best_overall is not None and time.time() < deadline - 1.0:
        col = best_overall.copy()
        col, final_score = local_search(col)
        if final_score < best_overall_score:
            best_overall_score = final_score
            best_overall = col

    return [int(c) for c in best_overall]
