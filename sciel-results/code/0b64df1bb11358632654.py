import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Aggressive safety margin

    N = 256

    # Precompute all 4-term arithmetic progressions (0-indexed)
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)

    # For each position, precompute the other 3 positions in each AP containing it
    aps_at = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            aps_at[pos].append(idx)

    # Build others_for_pos: for position i, array of shape (num_aps_containing_i, 3)
    others_for_pos = [None] * N
    for i in range(N):
        if not aps_at[i]:
            others_for_pos[i] = np.array([], dtype=np.int32).reshape(0, 3)
        else:
            others_list = []
            for ap_idx in aps_at[i]:
                row = aps[ap_idx]
                others = [p for p in row if p != i]
                others_list.append(others)
            others_for_pos[i] = np.array(others_list, dtype=np.int32)

    def count_mono(col):
        c = col[aps]
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))

    def fast_delta(col, i):
        """Change in mono count if we flip position i."""
        others = others_for_pos[i]
        if others.shape[0] == 0:
            return 0
        oc = col[others]
        all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
        if not np.any(all_same):
            return 0
        c_other = oc[:, 0]
        c_i = col[i]
        before = (c_other == c_i) & all_same
        after = (c_other == (1 - c_i)) & all_same
        return int(np.sum(after) - np.sum(before))

    def local_search(col, score, max_passes=50):
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 2.0:
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 2.0:
                    break
                d = fast_delta(col, i)
                if d < 0:
                    col[i] = 1 - col[i]
                    score += d
                    improved = True
        return col, score

    def generate_thue_morse():
        """Generate Thue-Morse sequence of length N."""
        tm = [0] * N
        for i in range(1, N):
            # tm[i] = tm[i // 2] ^ (i % 2)
            # Or simply popcount parity
            # Using a fast method:
            # tm[i] = 1 if popcount(i) is odd else 0
            # Python int.bit_count() is fast in 3.8+
            tm[i] = i.bit_count() % 2
        return np.array(tm, dtype=np.int32)

    def perturb(col, num_flips):
        """Create a perturbed copy of col by flipping num_flips random bits."""
        new_col = col.copy()
        indices = np.random.choice(N, num_flips, replace=False)
        new_col[indices] = 1 - new_col[indices]
        return new_col

    best_col = None
    best_score = float('inf')
    
    # Pre-generate some diverse starts
    thue_morse = generate_thue_morse()
    
    trial_count = 0
    while time.time() < deadline - 10.0:
        trial_count += 1
        # Alternate between new random/structured starts and perturbations of the best
        if best_col is None or trial_count % 3 != 0:
            # New start
            seed = 42 + trial_count * 97
            np_rng = np.random.RandomState(seed)
            
            r = np_rng.random()
            if r < 0.2:
                col = thue_morse.copy()
            elif r < 0.4:
                block = 1 + (trial_count % 8)
                col = np.array([(i // block) % 2 for i in range(N)], dtype=np.int32)
            elif r < 0.6:
                # Alternating blocks of size 2, 4, 8...
                col = np.array([(i >> (i & -i).bit_length() - 1) % 2 if i > 0 else 0 for i in range(N)], dtype=np.int32)
                # Simpler: just random
                col = np_rng.randint(0, 2, size=N).astype(np.int32)
            else:
                col = np_rng.randint(0, 2, size=N).astype(np.int32)
        else:
            # Perturb the best
            # Number of flips increases slightly over time or stays moderate
            num_flips = 5 + (trial_count % 10)
            col = perturb(best_col, num_flips)

        score = count_mono(col)
        col, score = local_search(col, score, max_passes=30)

        if score < best_score:
            best_score = score
            best_col = col.copy()

    # Final polish
    if best_col is not None and time.time() < deadline - 2.0:
        # Try a few more perturbation-refinement cycles on the best
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            # Small perturbation
            num_flips = 3
            col = perturb(best_col, num_flips)
            score = count_mono(col)
            col, score = local_search(col, score, max_passes=50)
            if score < best_score:
                best_score = score
                best_col = col.copy()
            else:
                # If no improvement, try larger perturbation
                num_flips = 10
                col = perturb(best_col, num_flips)
                score = count_mono(col)
                col, score = local_search(col, score, max_passes=50)
                if score < best_score:
                    best_score = score
                    best_col = col.copy()

    return [int(c) for c in best_col]
