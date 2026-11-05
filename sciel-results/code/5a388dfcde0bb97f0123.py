import time
import random
import numpy as np


def _max_ap_discrepancy(signs, n=256, limit=None):
    """Compute max absolute signed sum over any arithmetic progression in [0..n-1].
    O(N^2) scan over all (d, r) pairs.
    """
    max_abs = 1
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            min_p = 0
            max_p = 0
            cur = 0
            idx = r
            while idx < n:
                cur += signs[idx]
                if cur < min_p:
                    min_p = cur
                if cur > max_p:
                    max_p = cur
                idx += d
            disc = max_p - min_p
            if disc > max_abs:
                max_abs = disc
                if limit is not None and max_abs >= limit:
                    return max_abs
    return max_abs


def _find_worst_ap(signs, n=256):
    """Find the AP with the largest absolute discrepancy.
    Returns (d, r, disc, best_subarray) where best_subarray is (start_idx, end_idx) in the AP.
    """
    worst_disc = 0
    worst_d = 1
    worst_r = 0
    worst_start = 0
    worst_end = 0

    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            min_p = 0
            max_p = 0
            cur = 0
            idx = r
            min_p_idx = 0
            max_p_idx = 0
            ap_pos = 0
            while idx < n:
                cur += signs[idx]
                if cur < min_p:
                    min_p = cur
                    min_p_idx = ap_pos
                if cur > max_p:
                    max_p = cur
                    max_p_idx = ap_pos
                idx += d
                ap_pos += 1
            disc = max_p - min_p
            if disc > worst_disc:
                worst_disc = disc
                worst_d = d
                worst_r = r
                if min_p_idx < max_p_idx:
                    worst_start = min_p_idx
                    worst_end = max_p_idx
                else:
                    worst_start = max_p_idx
                    worst_end = min_p_idx
    return worst_d, worst_r, worst_disc, worst_start, worst_end


def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]


def _van_der_corput(n):
    """Generate van der Corput-based sign sequence."""
    signs = []
    for i in range(n):
        # Use bit-reversal to create a low-discrepancy pattern
        bits = bin(i)[2:].zfill(8)
        rev = int(bits[::-1], 2)
        signs.append(1 if rev % 2 == 0 else -1)
    return signs


def _multi_flip_search(signs, deadline, rng, n=256, max_flips=3):
    """Local search that allows flipping 2-3 signs at once to escape local minima."""
    current_signs = signs[:]
    current_val = _max_ap_discrepancy(current_signs, n)
    best_signs = current_signs[:]
    best_val = current_val

    temp = 0.8
    cool_rate = 0.9998
    steps = 0

    while time.time() < deadline - 3:
        # Choose how many flips to try
        num_flips = rng.randint(1, max_flips)
        
        # Choose indices to flip
        if num_flips == 1:
            flip_indices = [rng.randint(0, n - 1)]
        else:
            flip_indices = rng.sample(range(n), num_flips)
        
        old_vals = [current_signs[i] for i in flip_indices]
        for idx in flip_indices:
            current_signs[idx] = -current_signs[idx]

        new_val = _max_ap_discrepancy(current_signs, n, limit=best_val + 1)

        delta = new_val - current_val
        if delta < 0 or (delta == 0 and rng.random() < 0.3) or \
           (delta > 0 and rng.random() < np.exp(-delta / max(temp, 0.01))):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_signs = current_signs[:]
        else:
            for idx, ov in zip(flip_indices, old_vals):
                current_signs[idx] = ov

        temp *= cool_rate
        steps += 1

        if best_val == 1:
            break

    return best_signs, best_val, steps


def _targeted_multi_flip(signs, deadline, rng, n=256):
    """Targeted multi-flip: find worst APs, try flipping 2-3 elements within them."""
    current_signs = signs[:]
    current_val = _max_ap_discrepancy(current_signs, n)
    best_signs = current_signs[:]
    best_val = current_val

    rounds = 0
    while time.time() < deadline - 3:
        rounds += 1
        d, r, disc, start_pos, end_pos = _find_worst_ap(current_signs, n)
        if disc <= best_val:
            break

        ap_indices = list(range(r, n, d))
        if len(ap_indices) < 2:
            break

        subarray_indices = ap_indices[start_pos:end_pos + 1]
        if len(subarray_indices) < 2:
            break

        # Try all pairs of flips in the subarray
        best_pair_flip = None
        best_pair_val = current_val

        if len(subarray_indices) <= 15:
            for i in range(len(subarray_indices)):
                for j in range(i + 1, len(subarray_indices)):
                    idx1, idx2 = subarray_indices[i], subarray_indices[j]
                    current_signs[idx1] = -current_signs[idx1]
                    current_signs[idx2] = -current_signs[idx2]
                    v = _max_ap_discrepancy(current_signs, n, limit=best_pair_val)
                    if v < best_pair_val:
                        best_pair_val = v
                        best_pair_flip = (idx1, idx2)
                    current_signs[idx2] = -current_signs[idx2]
                    current_signs[idx1] = -current_signs[idx1]

        # Also try triplets if subarray is small enough
        if len(subarray_indices) <= 10:
            for i in range(len(subarray_indices)):
                for j in range(i + 1, len(subarray_indices)):
                    for k in range(j + 1, len(subarray_indices)):
                        idx1, idx2, idx3 = subarray_indices[i], subarray_indices[j], subarray_indices[k]
                        current_signs[idx1] = -current_signs[idx1]
                        current_signs[idx2] = -current_signs[idx2]
                        current_signs[idx3] = -current_signs[idx3]
                        v = _max_ap_discrepancy(current_signs, n, limit=best_pair_val)
                        if v < best_pair_val:
                            best_pair_val = v
                            best_pair_flip = (idx1, idx2, idx3)
                        current_signs[idx3] = -current_signs[idx3]
                        current_signs[idx2] = -current_signs[idx2]
                        current_signs[idx1] = -current_signs[idx1]

        if best_pair_flip is not None:
            for idx in best_pair_flip:
                current_signs[idx] = -current_signs[idx]
            current_val = best_pair_val
            if current_val < best_val:
                best_val = current_val
                best_signs = current_signs[:]
        else:
            # Try random multi-flips in the AP
            improved = False
            for _ in range(20):
                num_f = rng.randint(2, min(3, len(subarray_indices)))
                flip_set = rng.sample(subarray_indices, num_f)
                for idx in flip_set:
                    current_signs[idx] = -current_signs[idx]
                v = _max_ap_discrepancy(current_signs, n, limit=best_val + 1)
                if v < best_val:
                    best_val = v
                    best_signs = current_signs[:]
                    improved = True
                else:
                    for idx in flip_set:
                        current_signs[idx] = -current_signs[idx]
            if not improved and rounds > 5:
                break

    return best_signs, best_val


def _greedy_repair(signs, deadline, n=256, max_rounds=50):
    """Greedy repair: find worst AP, try flipping elements to reduce its discrepancy."""
    current_signs = signs[:]
    current_val = _max_ap_discrepancy(current_signs, n)
    best_signs = current_signs[:]
    best_val = current_val

    for _ in range(max_rounds):
        if time.time() > deadline - 3:
            break

        d, r, disc, start_pos, end_pos = _find_worst_ap(current_signs, n)
        if disc <= best_val:
            break

        ap_indices = list(range(r, n, d))
        if len(ap_indices) < 2:
            break

        subarray_indices = ap_indices[start_pos:end_pos + 1]

        best_flip = -1
        best_flip_val = current_val

        for idx in subarray_indices:
            current_signs[idx] = -current_signs[idx]
            v = _max_ap_discrepancy(current_signs, n, limit=best_flip_val)
            if v < best_flip_val:
                best_flip_val = v
                best_flip = idx
            current_signs[idx] = -current_signs[idx]

        if best_flip != -1:
            current_signs[best_flip] = -current_signs[best_flip]
            current_val = best_flip_val
            if current_val < best_val:
                best_val = current_val
                best_signs = current_signs[:]
        else:
            break

    return best_signs, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    n = 256
    rng = random.Random(42)

    # Phase 1: Generate multiple initial candidates
    candidates = []

    # Thue-Morse
    candidates.append(_thue_morse(n))

    # Van der Corput
    candidates.append(_van_der_corput(n))

    # Random initializations with various seeds
    for seed in range(8):
        rng2 = random.Random(seed * 1000 + 7)
        candidates.append([rng2.choice((-1, 1)) for _ in range(n)])

    # Evaluate candidates
    scored = []
    for c in candidates:
        v = _max_ap_discrepancy(c, n)
        scored.append((v, c))
    scored.sort(key=lambda x: x[0])

    best_signs = scored[0][1][:]
    best_val = scored[0][0]

    # Phase 2: Multi-flip local search on best candidates
    for v, c in scored[:4]:
        if time.time() > deadline - 25:
            break
        ls_signs, ls_val, _ = _multi_flip_search(c, deadline, rng, n, max_flips=3)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs

    # Phase 3: Targeted multi-flip on worst APs
    if time.time() < deadline - 10:
        tm_signs, tm_val = _targeted_multi_flip(best_signs, deadline, rng, n)
        if tm_val < best_val:
            best_val = tm_val
            best_signs = tm_signs

    # Phase 4: Greedy repair
    if time.time() < deadline - 5:
        gr_signs, gr_val = _greedy_repair(best_signs, deadline, n)
        if gr_val < best_val:
            best_val = gr_val
            best_signs = gr_signs

    # Phase 5: Final polish with multi-flip
    if time.time() < deadline - 2 and best_val > 1:
        ls_signs, ls_val, _ = _multi_flip_search(best_signs, deadline, rng, n, max_flips=2)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs

    # Phase 6: Another targeted multi-flip pass
    if time.time() < deadline - 1 and best_val > 1:
        tm_signs, tm_val = _targeted_multi_flip(best_signs, deadline, rng, n)
        if tm_val < best_val:
            best_val = tm_val
            best_signs = tm_signs

    return best_signs
