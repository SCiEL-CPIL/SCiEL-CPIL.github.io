import time
import random
import numpy as np
import math


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


def _van_der_corput(n):
    """Generate van der Corput-based sign sequence.
    Uses fractional parts of 1/b * i to create low-discrepancy ordering,
    then assigns signs based on position in sorted order.
    """
    # Generate van der Corput numbers
    vdc = []
    for i in range(n):
        x = 0.0
        d = 1.0
        ii = i
        while ii > 0:
            d /= 2.0
            x += (ii % 2) * d
            ii //= 2
        vdc.append(x)
    # Sort indices by van der Corput value, assign alternating signs
    order = sorted(range(n), key=lambda i: vdc[i])
    signs = [0] * n
    for pos, idx in enumerate(order):
        signs[idx] = 1 if pos % 2 == 0 else -1
    return signs


def _rademacher_orthogonal(n):
    """Generate sign sequence using Walsh-Rademacher functions (first n of them).
    This gives good low-discrepancy properties on structured progressions.
    """
    # Use the first non-trivial Walsh functions combined
    # Simple approach: use bit-reversal based signs
    signs = []
    for i in range(n):
        # Count bits in reversed position (rademacher-like)
        x = i
        cnt = 0
        while x > 0:
            cnt += x & 1
            x >>= 1
        # Use a different bit pattern than Thue-Morse for diversity
        # XOR with shifted version
        y = i ^ (i >> 3)
        cnt2 = 0
        while y > 0:
            cnt2 += y & 1
            y >>= 1
        signs.append(1 if (cnt + cnt2) % 2 == 0 else -1)
    return signs


def _local_search(signs, deadline, rng, n=256):
    """Simulated annealing local search with fast evaluation and 2-flip moves."""
    current_signs = signs[:]
    current_val = _max_ap_discrepancy(current_signs, n)
    best_signs = current_signs[:]
    best_val = current_val

    temp = 3.0
    cool_rate = 0.9992
    steps = 0
    stall_count = 0
    max_stall = 5000

    while time.time() < deadline - 2:
        # Occasionally use 2-flip moves to escape local minima
        if rng.random() < 0.15 and n >= 2:
            # 2-flip move: flip two random positions
            idx1 = rng.randint(0, n - 1)
            idx2 = rng.randint(0, n - 1)
            if idx1 == idx2:
                idx2 = (idx2 + 1) % n
            old1 = current_signs[idx1]
            old2 = current_signs[idx2]
            current_signs[idx1] = -old1
            current_signs[idx2] = -old2

            new_val = _max_ap_discrepancy(current_signs, n, limit=best_val + 1)
            delta = new_val - current_val

            if delta < 0 or (delta == 0 and rng.random() < 0.5) or \
               (delta > 0 and rng.random() < math.exp(-delta / max(temp, 0.01))):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_signs = current_signs[:]
                    stall_count = 0
            else:
                current_signs[idx1] = old1
                current_signs[idx2] = old2
                stall_count += 1
        else:
            # Single flip move
            idx = rng.randint(0, n - 1)
            old_val = current_signs[idx]
            current_signs[idx] = -old_val

            new_val = _max_ap_discrepancy(current_signs, n, limit=best_val + 1)
            delta = new_val - current_val

            if delta < 0 or (delta == 0 and rng.random() < 0.5) or \
               (delta > 0 and rng.random() < math.exp(-delta / max(temp, 0.01))):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_signs = current_signs[:]
                    stall_count = 0
            else:
                current_signs[idx] = old_val
                stall_count += 1

        temp *= cool_rate
        steps += 1

        # If stuck too long, restart temperature to escape local minimum
        if stall_count > max_stall:
            temp = max(temp, 2.0)
            stall_count = 0

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
                idx = random.choice(subarray_indices)
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

    # Van der Corput based
    candidates.append(_van_der_corput(n))

    # Rademacher-orthogonal variant
    candidates.append(_rademacher_orthogonal(n))

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
    # Allocate time proportional to remaining budget
    num_ls = min(4, len(scored))
    time_per_ls = (deadline - 15 - time.time()) / max(num_ls, 1)
    for i in range(num_ls):
        if time.time() > deadline - 15:
            break
        ls_deadline = min(deadline - 15, time.time() + time_per_ls)
        ls_signs, ls_val, _ = _local_search(scored[i][1], ls_deadline, rng, n)
        if ls_val < best_val:
            best_val = ls_val
            best_signs = ls_signs

    # Phase 3: Greedy repair
    if time.time() < deadline - 5 and best_val > 1:
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
