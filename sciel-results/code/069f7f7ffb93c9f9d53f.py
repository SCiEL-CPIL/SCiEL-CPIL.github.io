import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill

    # Precompute all APs: for each step d, each start s in [0, d)
    # AP = [s, s+d, s+2d, ...] while < n
    aps = []  # list of lists of indices
    index_to_aps = [[] for _ in range(n)]  # for each index, which APs contain it

    for d in range(1, n + 1):
        for s in range(d):
            idxs = list(range(s, n, d))
            if idxs:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)

    num_aps = len(aps)

    # Initialize signs randomly
    rng = random.Random(42)
    signs = [rng.choice([-1, 1]) for _ in range(n)]

    # Compute initial AP sums
    ap_sums = [0] * num_aps
    for ap_id, idxs in enumerate(aps):
        s = 0
        for idx in idxs:
            s += signs[idx]
        ap_sums[ap_id] = s

    # Build frequency array: freq[k] = number of APs with |ap_sums| == k
    # Max possible |sum| is n (all same sign), so size n+1
    max_possible = n
    freq = [0] * (max_possible + 1)
    for s in ap_sums:
        freq[abs(s)] += 1

    # Find current max discrepancy
    def get_max_disc():
        for k in range(max_possible, -1, -1):
            if freq[k] > 0:
                return k
        return 0

    current_max = get_max_disc()
    best_signs = signs[:]
    best_max = current_max

    # SA parameters
    T = 2.0
    T_min = 0.001
    cooling = 0.9995

    while time.time() - start_time < time_limit - 10.0:
        # Pick random index to flip
        i = rng.randrange(n)
        old_sign = signs[i]
        new_sign = -old_sign

        # Update AP sums for all APs containing i
        affected_aps = index_to_aps[i]
        old_abs_sums = []
        for ap_id in affected_aps:
            old_s = ap_sums[ap_id]
            old_abs = abs(old_s)
            old_abs_sums.append((ap_id, old_s, old_abs))
            freq[old_abs] -= 1
            new_s = old_s + 2 * new_sign  # flipping sign changes sum by 2*new_sign = -2*old_sign
            # Actually: sum changes by -2*old_sign = 2*new_sign
            ap_sums[ap_id] = new_s
            new_abs = abs(new_s)
            freq[new_abs] += 1

        new_max = get_max_disc()

        # Accept or reject
        if new_max <= current_max:
            # Accept (improvement or equal)
            signs[i] = new_sign
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs[:]
        else:
            # Reject: revert
            for ap_id, old_s, old_abs in old_abs_sums:
                freq[abs(ap_sums[ap_id])] -= 1
                ap_sums[ap_id] = old_s
                freq[old_abs] += 1
            # Don't change signs[i]
            # Boltzmann accept
            delta = new_max - current_max
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                signs[i] = new_sign
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_signs = signs[:]
                # Re-apply the flip properly
                for ap_id, old_s, old_abs in old_abs_sums:
                    freq[old_abs] -= 1
                    new_s = old_s + 2 * new_sign
                    ap_sums[ap_id] = new_s
                    freq[abs(new_s)] += 1
            else:
                # Revert was already done above
                pass

        T = max(T * cooling, T_min)

    # Local search refinement: greedy pass
    # Repeat until no improvement
    improved = True
    while improved and time.time() - start_time < time_limit - 2.0:
        improved = False
        for i in range(n):
            old_sign = signs[i]
            new_sign = -old_sign
            affected_aps = index_to_aps[i]
            old_abs_sums = []
            for ap_id in affected_aps:
                old_s = ap_sums[ap_id]
                old_abs = abs(old_s)
                old_abs_sums.append((ap_id, old_s, old_abs))
                freq[old_abs] -= 1
                new_s = old_s + 2 * new_sign
                ap_sums[ap_id] = new_s
                freq[abs(new_s)] += 1

            new_max = get_max_disc()

            if new_max < current_max:
                # Accept the flip
                signs[i] = new_sign
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs[:]
                improved = True
            else:
                # Revert
                for ap_id, old_s, old_abs in old_abs_sums:
                    freq[abs(ap_sums[ap_id])] -= 1
                    ap_sums[ap_id] = old_s
                    freq[old_abs] += 1

    return best_signs
