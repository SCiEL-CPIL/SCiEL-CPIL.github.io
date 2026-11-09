"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    d = np.asarray(demands, dtype=float)
    n = len(d)

    # Best-fit-decreasing style packing to identify good pairings.
    order = np.argsort(-d, kind="stable")
    bin_loads = []
    item_bin = np.full(n, -1, dtype=int)
    for idx in order:
        size = d[idx]
        # Best fit: choose bin with least remaining space that still fits.
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

    nbins = len(bin_loads)

    # Complementarity graph: reward pairs that leave residual capacity
    # as a near-integer multiple of the smallest item size in the instance.
    min_size = float(d.min())
    H = np.full((n, n), 1e-9, dtype=float)

    for bi in range(nbins):
        members = np.where(item_bin == bi)[0]
        if len(members) < 2:
            continue
        load = bin_loads[bi]
        tight = load / capacity  # in (0,1], higher means fuller bin

        for i in members:
            si = d[i]
            for j in members:
                if i == j:
                    continue
                sj = d[j]
                # Check if j fits after i in a bin of this load level.
                # After placing i, remaining space in a "tight" bin context:
                # We consider the residual after placing both i and j together.
                combined = si + sj
                if combined > capacity + 1e-12:
                    continue
                residual = capacity - combined
                if residual < 0:
                    continue

                # Structural fit: how close is residual to a multiple of min_size?
                # residual = k * min_size + r, where r is the "waste" that can't fit another min item.
                # We want r to be small (near-integer multiple) OR residual to be 0.
                if min_size > 0:
                    k = int(round(residual / min_size))
                    # Clamp k to reasonable range
                    k = max(0, k)
                    # The actual nearest multiple:
                    nearest_mult = k * min_size
                    # Distance to nearest multiple (could be k or k+1)
                    dist_to_k = abs(residual - k * min_size)
                    dist_to_k1 = abs(residual - (k + 1) * min_size)
                    min_dist = min(dist_to_k, dist_to_k1)
                    # Normalize: 0 means perfect structural fit
                    struct_score = 1.0 / (1.0 + min_dist / max(min_size, 1e-9))
                else:
                    struct_score = 1.0

                # Slack-based score: tighter is better (best-fit intuition)
                # residual = 0 is ideal; use inverse slack with smoothing
                slack = residual
                slack_score = 1.0 / (1.0 + slack / 10.0)

                # Size bonus: prefer larger items (fills more space)
                size_bonus = 0.5 + 0.5 * (sj / capacity)

                # Combine: structural fit is the primary signal,
                # slack and tightness modulate it.
                H[i, j] = tight * struct_score * slack_score * size_bonus

    # Ensure finite and positive floor.
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H
# EVOLVE-BLOCK-END
