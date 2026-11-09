# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    d = np.asarray(demands, dtype=float)
    n = len(d)

    # 1. Run Best-Fit-Decreasing (BFD) to get a reference packing
    order = np.argsort(-d, kind="stable")
    bin_loads = []
    item_bin = np.full(n, -1, dtype=int)
    for idx in order:
        size = d[idx]
        best_bi = -1
        best_rem = np.inf
        for bi in range(len(bin_loads)):
            rem = capacity - bin_loads[bi]
            if rem >= size and rem < best_rem:
                best_rem = rem
                best_bi = bi
        if best_bi >= 0:
            bin_loads[best_bi] += size
            item_bin[idx] = best_bi
        else:
            bin_loads.append(size)
            item_bin[idx] = len(bin_loads) - 1

    # Run a second reference: First-Fit-Decreasing (FFD)
    order2 = np.argsort(-d, kind="stable")
    bin_loads2 = []
    item_bin2 = np.full(n, -1, dtype=int)
    for idx in order2:
        size = d[idx]
        for bi in range(len(bin_loads2)):
            if capacity - bin_loads2[bi] >= size:
                bin_loads2[bi] += size
                item_bin2[idx] = bi
                break
        else:
            bin_loads2.append(size)
            item_bin2[idx] = len(bin_loads2) - 1

    # Use median item size as a more representative "fill unit" for complementarity
    med_size = float(np.median(d))
    if med_size < 1e-9:
        med_size = 1.0

    H = np.full((n, n), 1e-9, dtype=float)

    for i in range(n):
        si = d[i]
        bi_i = item_bin[i]
        bi2_i = item_bin2[i]

        for j in range(n):
            if i == j:
                continue
            sj = d[j]
            combined = si + sj
            if combined > capacity + 1e-12:
                continue

            residual = capacity - combined

            # Slack score: prefer tighter fits (lower residual)
            slack_score = 1.0 / (1.0 + residual / 10.0)

            # Complementarity score: residual close to k * med_size means the
            # remaining space can be filled with items of typical size.
            k = int(round(residual / med_size))
            k = max(0, k)
            dist_k = abs(residual - k * med_size)
            dist_k1 = abs(residual - (k + 1) * med_size)
            min_dist = min(dist_k, dist_k1)
            struct_score = 1.0 / (1.0 + min_dist / max(med_size, 1e-9))

            base_score = slack_score * struct_score

            # Consensus boost: if i and j are in the same bin in BOTH BFD and FFD,
            # that's a strong signal. Same in one only gives moderate boost.
            same_bfd = (bi_i == item_bin[j])
            same_ffd = (bi2_i == item_bin2[j])

            if same_bfd and same_ffd:
                H[i, j] = base_score * 4.0
            elif same_bfd or same_ffd:
                H[i, j] = base_score * 2.0
            else:
                H[i, j] = base_score * 0.7

    # Ensure finite and positive floor.
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H
# EVOLVE-BLOCK-END
