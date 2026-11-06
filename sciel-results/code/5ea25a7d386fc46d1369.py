import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s

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

    # Precompute a combined "others" matrix for all positions for fast vectorized delta
    # all_others[i] has shape (k_i, 3)
    all_others = np.array([others_for_pos[i] for i in range(N)], dtype=object)

    def count_mono(col):
        c = col[aps]
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))

    def all_deltas(col):
        """Vectorized: compute delta (score change) for flipping each position i."""
        deltas = np.zeros(N, dtype=np.int32)
        for i in range(N):
            others = others_for_pos[i]
            if others.shape[0] == 0:
                continue
            oc = col[others]  # shape (k, 3)
            all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
            if np.any(all_same):
                c_other = oc[:, 0]
                c_i = col[i]
                before = (c_other == c_i) & all_same
                after = (c_other == (1 - c_i)) & all_same
                deltas[i] = int(np.sum(after) - np.sum(before))
        return deltas

    def local_search_fast(col, score, max_passes=200):
        """Fast local search using precomputed deltas, multiple passes until convergence."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline - 2.0:
                break
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 2.0:
                    break
                others = others_for_pos[i]
                if others.shape[0] == 0:
                    continue
                oc = col[others]
                all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
                if not np.any(all_same):
                    continue
                c_other = oc[:, 0]
                c_i = col[i]
                before = (c_other == c_i) & all_same
                after = (c_other == (1 - c_i)) & all_same
                d = int(np.sum(after) - np.sum(before))
                if d < 0:
                    col[i] = 1 - col[i]
                    score += d
                    improved = True
        return col, score

    def smooth_surrogate(col, temperature=1.0):
        """Smooth surrogate via soft-min of per-AP monochromaticity.
        For each AP, compute a 'mono score' in [0,1] and sum with soft weighting.
        Lower temperature = sharper (closer to hard count)."""
        c = col[aps].astype(np.float64)
        # For each AP, mono score = 1 if all 4 same, else partial
        # Use: 1 - (number of distinct values - 1) / 3, clamped
        # Simpler: product of pairwise agreements
        # Use log-sum-exp style: for each AP, compute agreement count
        # Mono if all 4 equal. Partial credit: (agreements)/6 where agreements = sum of 6 pairwise equals
        a, b, ccol, d = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        # Pairwise agreements (6 pairs)
        ag = (a == b).astype(np.float64) + (a == ccol).astype(np.float64) + (a == d).astype(np.float64) \
           + (b == ccol).astype(np.float64) + (b == d).astype(np.float64) + (ccol == d).astype(np.float64)
        # ag in [0,6], mono means ag=6
        # Soft mono score: exp(-(6-ag)/temperature)
        scores = np.exp(-(6.0 - ag) / temperature)
        return float(np.sum(scores))

    def smooth_local_search(col, score, max_passes=50, temperature=0.5):
        """Local search using smooth surrogate to guide moves, then verify with hard count."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline - 2.0:
                break
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 2.0:
                    break
                others = others_for_pos[i]
                if others.shape[0] == 0:
                    continue
                oc = col[others]
                all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
                if not np.any(all_same):
                    continue
                c_other = oc[:, 0]
                c_i = col[i]
                before = (c_other == c_i) & all_same
                after = (c_other == (1 - c_i)) & all_same
                d = int(np.sum(after) - np.sum(before))
                if d < 0:
                    col[i] = 1 - col[i]
                    score += d
                    improved = True
        return col, score

    def sa_run(seed, max_steps, t0, t1):
        rng = random.Random(seed)
        col = np.random.RandomState(seed).randint(0, 2, size=N).astype(np.int32)
        cur = count_mono(col)
        best = col.copy()
        best_s = cur
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            t = t0 * (t1 / t0) ** (step / max_steps)
            i = rng.randint(0, N - 1)
            others = others_for_pos[i]
            if others.shape[0] == 0:
                continue
            oc = col[others]
            all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
            if not np.any(all_same):
                d = 0
            else:
                c_other = oc[:, 0]
                c_i = col[i]
                before = (c_other == c_i) & all_same
                after = (c_other == (1 - c_i)) & all_same
                d = int(np.sum(after) - np.sum(before))
            if d <= 0 or (t > 1e-12 and rng.random() < np.exp(-d / t)):
                col[i] = 1 - col[i]
                cur += d
                if cur < best_s:
                    best_s = cur
                    best = col.copy()
        return best, best_s

    def multi_flip_escape(col, score, num_flips, max_rounds=100):
        """Try flipping multiple bits simultaneously to escape local minima."""
        rng = random.Random(int(time.time() * 1000) % 100000)
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            indices = rng.sample(range(N), num_flips)
            new_col = col.copy()
            for i in indices:
                new_col[i] = 1 - new_col[i]
            new_score = count_mono(new_col)
            delta = new_score - score
            if delta < 0 or rng.random() < np.exp(-delta / 3.0):
                col = new_col
                score = new_score
        return col, score

    best_col = None
    best_score = float('inf')

    # Phase 1: Multi-start with diverse initializations + fast local search
    for trial in range(40):
        if time.time() > deadline - 40.0:
            break
        seed = 42 + trial * 97
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        # Diverse initializations
        r = rng.random()
        if r < 0.3:
            col = np_rng.randint(0, 2, size=N).astype(np.int32)
        elif r < 0.5:
            block = 1 + rng.randint(1, 12)
            col = np.array([(i // block) % 2 for i in range(N)], dtype=np.int32)
        elif r < 0.7:
            # Periodic with random period
            period = rng.randint(3, 20)
            col = np.array([(i % period) < period // 2 for i in range(N)], dtype=np.int32)
        elif r < 0.85:
            # Thue-Morse-like
            col = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32)
        else:
            # Random blocks of varying size
            col = np.zeros(N, dtype=np.int32)
            pos = 0
            while pos < N:
                length = rng.randint(1, 15)
                val = rng.randint(0, 1)
                col[pos:pos+length] = val
                pos += length

        score = count_mono(col)
        col, score = local_search_fast(col, score, max_passes=200)

        # Multi-flip escape for some trials
        if trial % 3 == 0 and time.time() < deadline - 30.0:
            col, score = multi_flip_escape(col, score, 8, max_rounds=50)
            col, score = local_search_fast(col, score, max_passes=100)

        if score < best_score:
            best_score = score
            best_col = col.copy()

    # Phase 2: SA runs
    for r in range(20):
        if time.time() > deadline - 20.0:
            break
        seed = 1000 + r * 211
        col, s = sa_run(seed, 80000, 3.0, 0.005)
        col, s = local_search_fast(col, s, max_passes=200)

        if time.time() < deadline - 15.0:
            col, s = multi_flip_escape(col, s, 10, max_rounds=40)
            col, s = local_search_fast(col, s, max_passes=100)

        if s < best_score:
            best_score = s
            best_col = col.copy()

    # Phase 3: Focused SA on best solution with perturbations
    if best_col is not None and time.time() < deadline - 10.0:
        for r in range(10):
            if time.time() > deadline - 5.0:
                break
            col = best_col.copy()
            # Perturb random subset
            num_pert = random.randint(5, 20)
            indices = random.sample(range(N), num_pert)
            for i in indices:
                col[i] = 1 - col[i]
            s = count_mono(col)
            col, s = local_search_fast(col, s, max_passes=200)
            if s < best_score:
                best_score = s
                best_col = col.copy()

    # Phase 4: Final polish
    if best_col is not None and time.time() < deadline - 2.0:
        col = best_col.copy()
        col, best_score = local_search_fast(col, best_score, max_passes=300)

        # Try smooth surrogate guided search
        if time.time() < deadline - 3.0:
            col2 = best_col.copy()
            col2, s2 = smooth_local_search(col2, best_score, max_passes=100, temperature=0.3)
            if s2 < best_score:
                best_score = s2
                best_col = col2

    return [int(c) for c in best_col]
