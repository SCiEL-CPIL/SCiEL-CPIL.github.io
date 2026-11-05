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
            min_p_idx = 0  # position in AP where min prefix occurs
            max_p_idx = 0  # position in AP where max prefix occurs
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
                # The worst subarray is between min_p_idx and max_p_idx (whichever comes first)
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


def _local_search(signs, deadline, rng, n=256):
    """Simulated annealing local search with fast evaluation."""
    current_signs = signs[:]
    current_val = _max_ap_discrepancy(current_signs, n)
    best_signs = current_signs[:]
    best_val = current_val

    temp = 1.0
    cool_rate = 0.9995
    steps = 0

    while time.time() < deadline - 2:
        idx = rng.randint(0, n - 1)
        old_val = current_signs[idx]
        current_signs[idx] = -old_val

        new_val = _max_ap_discrepancy(current_signs, n, limit=best_val + 1)

        delta = new_val - current_val
        if delta < 0 or (delta == 0 and rng.random() < 0.5) or \
           (delta > 0 and rng.random() < np.exp(-delta / max(temp, 0.01))):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_signs = current_signs[:]
        else:
            current_signs[idx] = old_val

        temp *= cool_rate
        steps += 1

        if best_val == 1:
            break

    return best_signs, best_val, steps


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

        # Get the elements of the worst AP
        ap_indices = list(range(r, n, d))
        if len(ap_indices) < 2:
            break

        # Try flipping each element in the worst subarray to see if it helps
        subarray_indices = ap_indices[start_pos:end_pos + 1]

        best_flip = -1
        best_flip_val = current_val

        for idx in subarray_indices:
            current_signs[idx] = -current_signs[idx]
            v = _max_ap_discrepancy(current_signs, n, limit=best_flip_val)
            if v < best_flip_val:
                best_flip_val = v
                best_flip = idx
            current_signs[idx] = -current_signs[idx]  # revert

        if best_flip != -1:
            current_signs[best_flip] = -current_signs[best_flip]
            current_val = best_flip_val
            if current_val < best_val:
                best_val = current_val
                best_signs = current_signs[:]
        else:
            # No single flip helps; try random flips in the AP
            improved = False
            for _ in range(10):
                idx = rng.choice(subarray_indices) if 'rng' in dir() else random.choice(subarray_indices)
                current_signs[idx] = -current_signs[idx]
                v = _max_ap_discrepancy(current_signs, n, limit=best_val + 1)
                if v < best_val:
                    best_val = v
                    best_signs = current_signs[:]
                    improved = True
                else:
                    current_signs[idx] = -current_signs[idx]  # revert
            if not improved:
                break

    return best_signs, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    n = 256
    rng = random.Random(42)

    # Phase 1: Generate multiple initial candidates
    candidates = []

    # Thue-Morse
    candidates.append(_thue_morse(n))

    # Random initializations
    for seed in range(6):
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

    # Phase 2: Local search on best candidates
    for v, c in scored[:3]:
        if time.time() > deadline - 15:
            break
        ls_signs, ls_val, _ = _local_search(c, deadline, rng, n)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs

    # Phase 3: Greedy repair
    if time.time() < deadline - 5:
        gr_signs, gr_val = _greedy_repair(best_signs, deadline, n)
        if gr_val < best_val:
            best_val = gr_val
            best_signs = gr_signs

    # Phase 4: Final polish - more local search
    if time.time() < deadline - 2 and best_val > 1:
        ls_signs, ls_val, _ = _local_search(best_signs, deadline, rng, n)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs

    # Phase 5: Another greedy repair pass
    if time.time() < deadline - 1 and best_val > 1:
        gr_signs, gr_val = _greedy_repair(best_signs, deadline, n)
        if gr_val < best_val:
            best_val = gr_val
            best_signs = gr_signs

    return best_signs
