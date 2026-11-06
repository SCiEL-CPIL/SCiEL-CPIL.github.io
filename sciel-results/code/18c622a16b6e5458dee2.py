import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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
        """Compute score change for flipping position i."""
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

    def local_search(col, score, max_passes=1000):
        """Fast local search using precomputed deltas."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline - 5.0:
                break
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                d = fast_delta(col, i)
                if d < 0:
                    col[i] = 1 - col[i]
                    score += d
                    improved = True
        return col, score

    def sa_run(seed, max_steps, t0, t1):
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        # Start with Thue-Morse or Random
        if seed % 2 == 0:
            col = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32)
        else:
            col = np_rng.randint(0, 2, size=N).astype(np.int32)
        
        cur = count_mono(col)
        best = col.copy()
        best_s = cur
        
        for step in range(max_steps):
            if time.time() > deadline - 10.0:
                break
            t = t0 * (t1 / t0) ** (step / max_steps)
            i = rng.randint(0, N - 1)
            d = fast_delta(col, i)
            if d <= 0 or (t > 1e-12 and rng.random() < np.exp(-d / t)):
                col[i] = 1 - col[i]
                cur += d
                if cur < best_s:
                    best_s = cur
                    best = col.copy()
        return best, best_s

    best_col = None
    best_score = float('inf')

    # Phase 1: Thue-Morse based local search
    thue_col = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32)
    thue_score = count_mono(thue_col)
    thue_col, thue_score = local_search(thue_col, thue_score, max_passes=500)
    if thue_score < best_score:
        best_score = thue_score
        best_col = thue_col.copy()

    # Phase 2: Random starts + SA
    for trial in range(20):
        if time.time() > deadline - 30.0:
            break
        seed = 100 + trial
        col, s = sa_run(seed, 50000, 5.0, 0.1)
        col, s = local_search(col, s, max_passes=500)
        if s < best_score:
            best_score = s
            best_col = col.copy()

    # Phase 3: Intensive polish on best
    if best_col is not None and time.time() < deadline - 10.0:
        col = best_col.copy()
        col, best_score = local_search(col, best_score, max_passes=2000)

    return [int(c) for c in best_col]
