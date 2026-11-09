import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 85.0

    # Precompute all APs
    aps = []
    index_to_aps = [[] for _ in range(n)]
    for d in range(1, n + 1):
        for s in range(d):
            idxs = list(range(s, n, d))
            if idxs:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)

    num_aps = len(aps)

    # Random initialization
    rng = random.Random(42)
    signs = [rng.choice([-1, 1]) for _ in range(n)]

    # Compute AP sums
    ap_sums = [0] * num_aps
    for ap_id, idxs in enumerate(aps):
        s = 0
        for idx in idxs:
            s += signs[idx]
        ap_sums[ap_id] = s

    def compute_max():
        m = 0
        for s in ap_sums:
            a = abs(s)
            if a > m:
                m = a
        return m

    current_max = compute_max()

    # Greedy initialization
    improved = True
    while improved and time.time() - start_time < time_limit - 35.0:
        improved = False
        for i in range(n):
            new_sign = -signs[i]
            for ap_id in index_to_aps[i]:
                ap_sums[ap_id] += 2 * new_sign
            new_max = compute_max()
            if new_max < current_max:
                signs[i] = new_sign
                current_max = new_max
                improved = True
            else:
                for ap_id in index_to_aps[i]:
                    ap_sums[ap_id] -= 2 * new_sign

    best_signs = signs[:]
    best_max = current_max

    # Multi-flip SA
    T = 3.0
    T_min = 0.01
    cooling = 0.9997

    while time.time() - start_time < time_limit - 5.0:
        k = rng.choice([2, 3, 4])
        indices = rng.sample(range(n), k)
        old_signs = [signs[i] for i in indices]
        new_signs = [-s for s in old_signs]

        # Apply
        for idx, ns in zip(indices, new_signs):
            for ap_id in index_to_aps[idx]:
                ap_sums[ap_id] += 2 * ns

        new_max = compute_max()

        if new_max <= current_max:
            for idx, ns in zip(indices, new_signs):
                signs[idx] = ns
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs[:]
        else:
            # Revert
            for j, idx in enumerate(indices):
                for ap_id in index_to_aps[idx]:
                    ap_sums[ap_id] -= 2 * new_signs[j]
                signs[idx] = old_signs[j]
            delta = new_max - current_max
            if rng.random() < math.exp(-delta / max(T, 1e-10)):
                for idx, ns in zip(indices, new_signs):
                    for ap_id in index_to_aps[idx]:
                        ap_sums[ap_id] += 2 * ns
                    signs[idx] = ns
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_signs = signs[:]

        T = max(T * cooling, T_min)

    # Final greedy
    improved = True
    while improved and time.time() - start_time < time_limit - 2.0:
        improved = False
        for i in range(n):
            new_sign = -signs[i]
            for ap_id in index_to_aps[i]:
                ap_sums[ap_id] += 2 * new_sign
            new_max = compute_max()
            if new_max < current_max:
                signs[i] = new_sign
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs[:]
            else:
                for ap_id in index_to_aps[i]:
                    ap_sums[ap_id] -= 2 * new_sign

    return best_signs
