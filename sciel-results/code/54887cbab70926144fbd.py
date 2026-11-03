"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0

    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    B_p = b[np.ix_(p, p)]
    current_cost = float(np.sum(a * B_p))
    best_cost = current_cost
    best_p = p.copy()

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    T = 500.0
    cooling = 0.9995
    T_min = 0.5
    batch_size = 50

    while time.time() < deadline and T > T_min:
        idx1 = np.random.randint(0, n, size=batch_size)
        idx2 = np.random.randint(0, n, size=batch_size)
        mask = idx1 != idx2
        if not np.all(mask):
            idx2[~mask] = np.random.randint(0, n, size=np.sum(~mask))
            mask = idx1 != idx2
            if not np.all(mask):
                idx1 = idx1[mask]
                idx2 = idx2[mask]

        i_arr = idx1
        j_arr = idx2
        loc_i = p[i_arr]
        loc_j = p[j_arr]

        deltas = np.zeros(len(i_arr), dtype=np.float64)
        for m in range(len(i_arr)):
            i = i_arr[m]
            j = j_arr[m]
            li = loc_i[m]
            lj = loc_j[m]

            cross = a_f[i, j] * (b_f[lj, li] - b_f[li, lj]) + a_f[j, i] * (b_f[li, lj] - b_f[lj, li])

            diff_i = b_f[lj, p] - b_f[li, p]
            sum_i = a_f[i, :].dot(diff_i) - a_f[i, i] * diff_i[i] - a_f[i, j] * diff_i[j]

            diff_i2 = b_f[p, lj] - b_f[p, li]
            sum_i2 = a_f[:, i].dot(diff_i2) - a_f[i, i] * diff_i2[i] - a_f[j, i] * diff_i2[j]

            diff_j = b_f[li, p] - b_f[lj, p]
            sum_j = a_f[j, :].dot(diff_j) - a_f[j, i] * diff_j[i] - a_f[j, j] * diff_j[j]

            diff_j2 = b_f[p, li] - b_f[p, lj]
            sum_j2 = a_f[:, j].dot(diff_j2) - a_f[i, j] * diff_j2[i] - a_f[j, j] * diff_j2[j]

            deltas[m] = cross + sum_i + sum_i2 + sum_j + sum_j2

        accept_mask = deltas < 0
        pos_mask = ~accept_mask
        if np.any(pos_mask):
            accept_prob = np.exp(-deltas[pos_mask] / T)
            rand = np.random.random(np.sum(pos_mask))
            accept_pos = rand < accept_prob
            pos_indices = np.where(pos_mask)[0]
            accept_mask[pos_indices[accept_pos]] = True

        accepted = np.where(accept_mask)[0]
        for m in accepted:
            i = i_arr[m]
            j = j_arr[m]
            p[i], p[j] = p[j], p[i]
            current_cost += deltas[m]

        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()

        T *= cooling

    return best_p.tolist()
# EVOLVE-BLOCK-END
