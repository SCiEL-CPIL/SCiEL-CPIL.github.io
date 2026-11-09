"""QUBO formulation with auxiliary variables for monochromatic 4-AP penalties,
solved via vectorized penalty annealing on the quadratic energy landscape."""
import numpy as np
import time

def construct():
    N = 256
    rng = np.random.default_rng(42)
    start_time = time.time()
    max_time = 115.0

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    aps_list = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            aps_list.append(idx)

    aps = np.array(aps_list, dtype=np.int32)
    num_aps = len(aps)

    # QUBO formulation:
    # For each 4-AP (i,j,k,l), introduce auxiliary binary variables:
    #   m0 = 1 if x_i=x_j=x_k=x_l=0 (all-zero mono)
    #   m1 = 1 if x_i=x_j=x_k=x_l=1 (all-one mono)
    #
    # The monochromatic penalty is m0 + m1.
    #
    # We use penalty terms to enforce:
    #   m0 <= (1-x_i) + (1-x_j) + (1-x_k) + (1-x_l) - 3
    #   m1 <= x_i + x_j + x_k + x_l - 3
    #
    # Equivalently, m0 = 1 only when all four are 0, m1 = 1 only when all four are 1.
    #
    # Objective: minimize sum over all APs of (m0 + m1) + lambda * penalty_violations
    #
    # Penalty for m0: if any of x_i..x_l is 1, m0 should be 0.
    #   P0 = m0 * (x_i + x_j + x_k + x_l)  -- penalizes m0=1 when any x=1
    #   Also need: if all x=0, m0 can be 0 or 1; we want m0=1 to count the AP.
    #   Actually, we want m0 to be 1 iff all x=0. So:
    #   P0 = m0 * (x_i + x_j + x_k + x_l) + (1 - m0) * (1 - (1-x_i)*(1-x_j)*(1-x_k)*(1-x_l))
    #
    # Simpler approach: use the fact that for binary variables,
    #   m0 = 1 iff all x=0  <=>  m0 <= prod(1-x) and m0 >= prod(1-x)
    #
    # For annealing, we work directly with the coloring x and compute the
    # effective energy as the number of monochromatic APs, but we use the
    # QUBO structure to compute gradients efficiently.

    # Initial random coloring
    x = rng.integers(0, 2, size=N).astype(np.int8)

    def count_mono_aps(x):
        vals = x[aps]
        return int(((vals[:, 0] == vals[:, 1]) &
                    (vals[:, 1] == vals[:, 2]) &
                    (vals[:, 2] == vals[:, 3])).sum())

    current_energy = count_mono_aps(x)
    best_x = x.copy()
    best_energy = current_energy

    # Precompute flattened indices for bincount
    var_indices_flat = aps.ravel()

    # Penalty annealing with QUBO-style energy landscape
    # Temperature schedule: start high, cool geometrically
    T = 20.0
    T_min = 0.01
    cooling = 0.9995
    batch_size = 32

    while T > T_min:
        if time.time() - start_time > max_time:
            break

        # Compute current values for all APs
        vals = x[aps]  # shape (num_aps, 4)

        # For each position in each AP, compute:
        #   delta_0[i] = change in energy if we flip x[i] from 0 to 1
        #   delta_1[i] = change in energy if we flip x[i] from 1 to 0
        #
        # Flipping x[i] from 0 to 1:
        #   For each AP containing i at position p:
        #     Before: if all 4 were 0, it was mono (contributes 1); after flip, not mono (contributes 0). Delta = -1.
        #     If not all 0 before: check if all 1 after. If so, delta = +1.
        #     Otherwise, delta = 0.
        #
        # We compute this vectorized:
        # For position p in AP, the "others" are the other 3 positions.
        # Flipping from 0 to 1:
        #   -1 if others are all 0 (was all-0 mono, now broken)
        #   +1 if others are all 1 (becomes all-1 mono)
        #   0 otherwise

        # Compute for each position
        # oeq0[p]: for APs, are the other 3 positions all 0?
        # oeq1[p]: for APs, are the other 3 positions all 1?

        # Position 0: others are 1,2,3
        oeq0_0 = (vals[:, 1] == 0) & (vals[:, 2] == 0) & (vals[:, 3] == 0)
        oeq1_0 = (vals[:, 1] == 1) & (vals[:, 2] == 1) & (vals[:, 3] == 1)
        # Position 1: others are 0,2,3
        oeq0_1 = (vals[:, 0] == 0) & (vals[:, 2] == 0) & (vals[:, 3] == 0)
        oeq1_1 = (vals[:, 0] == 1) & (vals[:, 2] == 1) & (vals[:, 3] == 1)
        # Position 2: others are 0,1,3
        oeq0_2 = (vals[:, 0] == 0) & (vals[:, 1] == 0) & (vals[:, 3] == 0)
        oeq1_2 = (vals[:, 0] == 1) & (vals[:, 1] == 1) & (vals[:, 3] == 1)
        # Position 3: others are 0,1,2
        oeq0_3 = (vals[:, 0] == 0) & (vals[:, 1] == 0) & (vals[:, 2] == 0)
        oeq1_3 = (vals[:, 0] == 1) & (vals[:, 1] == 1) & (vals[:, 2] == 1)

        # For flipping from 0 to 1: delta = -1*oeq0 + 1*oeq1 (per AP-position)
        # For flipping from 1 to 0: delta = +1*oeq1 - 1*oeq0 (per AP-position)
        # Wait, let me re-derive:
        # Flip 0->1 at position p in an AP:
        #   If others all 0: was all-0 mono (counted 1), now broken (counted 0). Delta = -1.
        #   If others all 1: was not mono (others are 1, this was 0), now all-1 mono (counted 1). Delta = +1.
        #   Else: no change. Delta = 0.
        # So delta_0to1 = oeq1 - oeq0 (per AP at that position)
        #
        # Flip 1->0 at position p in an AP:
        #   If others all 1: was all-1 mono (counted 1), now broken (counted 0). Delta = -1.
        #   If others all 0: was not mono, now all-0 mono (counted 1). Delta = +1.
        #   Else: no change. Delta = 0.
        # So delta_1to0 = oeq0 - oeq1 (per AP at that position)

        # Stack into arrays of shape (num_aps, 4)
        oeq0 = np.stack([oeq0_0, oeq0_1, oeq0_2, oeq0_3], axis=1).astype(np.float32)
        oeq1 = np.stack([oeq1_0, oeq1_1, oeq1_2, oeq1_3], axis=1).astype(np.float32)

        # delta_0to1[ap, pos] = oeq1 - oeq0
        # delta_1to0[ap, pos] = oeq0 - oeq1
        delta_0to1 = oeq1 - oeq0
        delta_1to0 = oeq0 - oeq1

        # Sum over all APs for each variable
        # For each variable i, delta_if_0[i] = sum of delta_0to1 over all AP-positions where var i appears
        # For each variable i, delta_if_1[i] = sum of delta_1to0 over all AP-positions where var i appears

        delta_if_0 = np.bincount(var_indices_flat, weights=delta_0to1.ravel(), minlength=N)
        delta_if_1 = np.bincount(var_indices_flat, weights=delta_1to0.ravel(), minlength=N)

        # Select delta based on current x
        delta_all = np.where(x == 0, delta_if_0, delta_if_1)

        # Select a batch of variables to consider flipping
        indices = rng.integers(0, N, size=batch_size)

        # Sort by delta to prioritize improving moves
        deltas = delta_all[indices]
        order = np.argsort(deltas)

        for idx in order:
            i = indices[idx]
            d = deltas[idx]

            if d < 0:
                # Always accept improving moves
                x[i] = 1 - x[i]
                current_energy += d
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
            elif d == 0:
                # Accept neutral moves with some probability to explore
                if rng.random() < 0.5:
                    x[i] = 1 - x[i]
            else:
                # Worsening move: accept with Metropolis probability
                if d < T * 4:
                    if rng.random() < np.exp(-d / T):
                        x[i] = 1 - x[i]
                        current_energy += d

        T *= cooling

    # Final local search polish
    for _ in range(5000):
        if time.time() - start_time > max_time:
            break
        improved = False
        for i in range(N):
            # Compute delta for flipping i
            vals = x[aps]
            # Recompute deltas for this variable only
            # Find all APs containing i
            mask = np.any(aps == i, axis=1)
            if not np.any(mask):
                continue
            sub_aps = aps[mask]
            sub_vals = x[sub_aps]
            # For each AP, find position of i
            pos_in_ap = np.where(sub_aps == i, 1, 0).argmax(axis=1)
            # Compute others
            others = sub_vals.copy()
            others[np.arange(len(sub_vals)), pos_in_ap] = 1 - x[i]
            # Check if all equal after flip
            all_eq = (others[:, 0] == others[:, 1]) & (others[:, 1] == others[:, 2]) & (others[:, 2] == others[:, 3])
            # Current: check if all equal now
            cur_eq = (sub_vals[:, 0] == sub_vals[:, 1]) & (sub_vals[:, 1] == sub_vals[:, 2]) & (sub_vals[:, 2] == sub_vals[:, 3])
            delta = all_eq.sum() - cur_eq.sum()
            if delta < 0:
                x[i] = 1 - x[i]
                current_energy += delta
                improved = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
        if not improved:
            break

    return [int(v) for v in best_x]
