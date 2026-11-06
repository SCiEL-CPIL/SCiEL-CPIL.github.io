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

    def local_search(col, score, max_passes=300):
        """Fast local search using precomputed deltas."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline - 3.0:
                break
            improved = False
            passes += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 3.0:
                    break
                d = fast_delta(col, i)
                if d < 0:
                    col[i] = 1 - col[i]
                    score += d
                    improved = True
        return col, score

    def vectorized_multi_flip(col, score, num_flips, max_rounds=50, rng=None):
        """Try flipping multiple bits simultaneously to escape local minima."""
        if rng is None:
            rng = random.Random(int(time.time() * 1000) % 100000)
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            indices = rng.sample(range(N), num_flips)
            new_col = col.copy()
            for i in indices:
                new_col[i] = 1 - new_col[i]
            new_score = count_mono(new_col)
            delta = new_score - score
            if delta < 0 or (delta < 5 and rng.random() < np.exp(-delta / 4.0)):
                col = new_col
                score = new_score
        return col, score

    def sa_run(seed, max_steps, t0, t1):
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        col = np_rng.randint(0, 2, size=N).astype(np.int32)
        cur = count_mono(col)
        best = col.copy()
        best_s = cur
        for step in range(max_steps):
            if time.time() > deadline - 8.0:
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

    def high_temp_anneal(col, score, max_steps=50000, t0=10.0, t1=0.1):
        """High-temperature annealing to escape deep local minima."""
        rng = random.Random(int(time.time() * 1000) % 100000)
        cur = score
        best = col.copy()
        best_s = score
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

    def subgradient_step(col, score, lr=0.5, max_iter=200):
        """Subgradient-style update: for each position, compute the 'pressure'
        from APs that would become monochromatic if flipped, and bias flips accordingly."""
        if time.time() > deadline - 5.0:
            return col, score
        improved = True
        iters = 0
        while improved and iters < max_iter:
            if time.time() > deadline - 5.0:
                break
            improved = False
            iters += 1
            order = list(range(N))
            random.shuffle(order)
            for i in order:
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
                elif d == 0 and random.random() < 0.1:
                    # Random walk for flat moves
                    col[i] = 1 - col[i]
                    improved = True
        return col, score

    best_col = None
    best_score = float('inf')

    # Phase 1: Multi-start with diverse initializations + local search
    for trial in range(50):
        if time.time() > deadline - 50.0:
            break
        seed = 42 + trial * 97
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        # Diverse initializations
        r = rng.random()
        if r < 0.25:
            col = np_rng.randint(0, 2, size=N).astype(np.int32)
        elif r < 0.4:
            block = 1 + rng.randint(1, 12)
            col = np.array([(i // block) % 2 for i in range(N)], dtype=np.int32)
        elif r < 0.55:
            period = rng.randint(3, 20)
            col = np.array([(i % period) < period // 2 for i in range(N)], dtype=np.int32)
        elif r < 0.7:
            # Thue-Morse-like
            col = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32)
        elif r < 0.85:
            # Random blocks of varying size
            col = np.zeros(N, dtype=np.int32)
            pos = 0
            while pos < N:
                length = rng.randint(1, 15)
                val = rng.randint(0, 1)
                col[pos:pos+length] = val
                pos += length
        else:
            # Checkerboard with noise
            col = np.array([(i % 2) for i in range(N)], dtype=np.int32)
            noise_mask = np_rng.random(N) < 0.15
            col[noise_mask] = 1 - col[noise_mask]

        score = count_mono(col)
        col, score = local_search(col, score, max_passes=300)

        # Multi-flip escape for some trials
        if trial % 2 == 0 and time.time() < deadline - 35.0:
            col, score = vectorized_multi_flip(col, score, 8, max_rounds=40, rng=rng)
            col, score = local_search(col, score, max_passes=150)

        if score < best_score:
            best_score = score
            best_col = col.copy()

    # Phase 2: SA runs with better cooling
    for r in range(25):
        if time.time() > deadline - 25.0:
            break
        seed = 1000 + r * 211
        col, s = sa_run(seed, 100000, 5.0, 0.05)
        col, s = local_search(col, s, max_passes=300)

        if time.time() < deadline - 18.0:
            col, s = vectorized_multi_flip(col, s, 10, max_rounds=30)
            col, s = local_search(col, s, max_passes=150)

        if s < best_score:
            best_score = s
            best_col = col.copy()

    # Phase 3: High-temp annealing on best solution
    if best_col is not None and time.time() < deadline - 15.0:
        for r in range(8):
            if time.time() > deadline - 8.0:
                break
            col = best_col.copy()
            col, s = high_temp_anneal(col, best_score, max_steps=30000, t0=8.0, t1=0.2)
            col, s = local_search(col, s, max_passes=200)
            if s < best_score:
                best_score = s
                best_col = col.copy()

    # Phase 4: Focused perturbations on best solution
    if best_col is not None and time.time() < deadline - 8.0:
        for r in range(15):
            if time.time() > deadline - 4.0:
                break
            col = best_col.copy()
            num_pert = random.randint(5, 25)
            indices = random.sample(range(N), num_pert)
            for i in indices:
                col[i] = 1 - col[i]
            s = count_mono(col)
            col, s = local_search(col, s, max_passes=300)
            if s < best_score:
                best_score = s
                best_col = col.copy()

    # Phase 5: Subgradient polish
    if best_col is not None and time.time() < deadline - 4.0:
        col = best_col.copy()
        col, best_score = subgradient_step(col, best_score, lr=0.5, max_iter=100)
        col, best_score = local_search(col, best_score, max_passes=300)

    # Phase 6: Final polish
    if best_col is not None and time.time() < deadline - 2.0:
        col = best_col.copy()
        col, best_score = local_search(col, best_score, max_passes=500)

    return [int(c) for c in best_col]
