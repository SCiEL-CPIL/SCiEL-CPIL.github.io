import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Leave a safety margin

    # Generate all APs within [0, n)
    aps = []
    for d in range(1, n):
        for a in range(d):
            seq = list(range(a, n, d))
            if len(seq) >= 2:
                aps.append(seq)

    m = len(aps)

    # Build membership matrix: member[j, i] = 1 if index i is in AP j
    member = np.zeros((m, n), dtype=np.int8)
    for j, ap in enumerate(aps):
        for idx in ap:
            member[j, idx] = 1

    # Precompute affected AP indices for each sign position
    # This avoids calling np.nonzero in the inner loop
    affected_list = []
    for i in range(n):
        affected_list.append(np.nonzero(member[:, i])[0])

    # Greedy initialization: assign signs one by one to minimize local AP conflicts
    signs = np.zeros(n, dtype=np.int8)
    ap_sums = np.zeros(m, dtype=np.int16)

    for i in range(n):
        best_sign = 1
        best_local_max = float('inf')
        affected = affected_list[i]
        if len(affected) == 0:
            signs[i] = 1
            continue
        for s in [1, -1]:
            new_sums = ap_sums[affected] + s
            local_max = np.max(np.abs(new_sums))
            if local_max < best_local_max:
                best_local_max = local_max
                best_sign = s
        signs[i] = best_sign
        ap_sums[affected] += best_sign

    current_max = int(np.max(np.abs(ap_sums)))
    best_signs = signs.copy()
    best_max = current_max

    # Local search: Hill Climbing with Simulated Annealing
    no_improve_count = 0
    max_no_improve = 40
    temperature = 2.0
    cooling = 0.97
    rng = np.random.default_rng(42)

    while time.time() < deadline - 5:
        improved = False
        indices = list(range(n))
        rng.shuffle(indices)

        for i in indices:
            if time.time() > deadline - 2:
                break

            old_sign = signs[i]
            new_sign = -old_sign
            signs[i] = new_sign

            affected = affected_list[i]
            if len(affected) > 0:
                ap_sums[affected] += 2 * new_sign

            new_max = int(np.max(np.abs(ap_sums)))

            # Simulated annealing: accept worse moves with some probability
            if new_max < current_max:
                current_max = new_max
                improved = True
                # Keep flip
            else:
                delta = new_max - current_max
                if delta > 0:
                    prob = np.exp(-delta / temperature)
                    if rng.random() < prob:
                        # Accept worse move
                        current_max = new_max
                        improved = True
                    else:
                        # Revert
                        if len(affected) > 0:
                            ap_sums[affected] -= 2 * new_sign
                        signs[i] = old_sign

        temperature *= cooling

        if improved:
            no_improve_count = 0
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reset temperature to escape local minima
                temperature = 2.0
                no_improve_count = 0
                # If we've been stuck for too long, break
                if temperature > 10:
                    break
                temperature = 2.0  # Reset to try again
                no_improve_count = 0

    return best_signs.tolist()
