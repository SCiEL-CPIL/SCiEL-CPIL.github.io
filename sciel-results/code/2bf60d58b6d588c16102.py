"""Quartet-informed neighbour-joining with deadline guard."""
import time
import math
import numpy as np


def _jc_dist_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using numpy."""
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(n_sites):
            mat[i, j] = code_map[seq[j]]
    diff = (mat[:, None, :] != mat[None, :, :]).astype(np.float64)
    mismatches = diff.sum(axis=2)
    p = mismatches / n_sites
    np.fill_diagonal(p, 0.0)
    p_clipped = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _quartet_informed_dist(dist):
    """
    Adjust the distance matrix using quartet voting.
    For each quartet (a,b,c,d), determine the best topology using the additive quartet test,
    then add a small correction to the sister-pair distances.
    """
    n = dist.shape[0]
    correction = np.zeros((n, n))
    alpha = 0.15  # correction strength

    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    d_ab = dist[a, b]
                    d_ac = dist[a, c]
                    d_ad = dist[a, d]
                    d_bc = dist[b, c]
                    d_bd = dist[b, d]
                    d_cd = dist[c, d]

                    # Internal edge length estimates for each topology
                    s_ab_cd = (d_ac + d_bd - d_ab - d_cd) / 2.0
                    s_ac_bd = (d_ab + d_cd - d_ac - d_bd) / 2.0
                    s_ad_bc = (d_ab + d_cd - d_ad - d_bc) / 2.0

                    scores = [s_ab_cd, s_ac_bd, s_ad_bc]
                    best = int(np.argmax(scores))

                    if best == 0:
                        val = max(0.0, s_ab_cd)
                        correction[a, b] += alpha * val
                        correction[b, a] += alpha * val
                        correction[c, d] += alpha * val
                        correction[d, c] += alpha * val
                    elif best == 1:
                        val = max(0.0, s_ac_bd)
                        correction[a, c] += alpha * val
                        correction[c, a] += alpha * val
                        correction[b, d] += alpha * val
                        correction[d, b] += alpha * val
                    else:
                        val = max(0.0, s_ad_bc)
                        correction[a, d] += alpha * val
                        correction[d, a] += alpha * val
                        correction[b, c] += alpha * val
                        correction[c, b] += alpha * val

    new_dist = dist - correction
    np.fill_diagonal(new_dist, 0.0)
    new_dist = np.maximum(new_dist, 0.0)
    return new_dist


def _nj_tree(dist, labels):
    """Standard neighbour joining. dist is (n,n) numpy array, labels is list of str."""
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 1:
        return labels[0]

    nodes = list(labels)
    trees = {name: name for name in labels}
    dist_dict = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = float(dist[i, j])
            dist_dict[(nodes[i], nodes[j])] = d
            dist_dict[(nodes[j], nodes[i])] = d

    while len(nodes) > 3:
        n_cur = len(nodes)
        row_sum = {}
        for a in nodes:
            s = 0.0
            for b in nodes:
                if a != b:
                    s += dist_dict[(a, b)]
            row_sum[a] = s

        best_criterion = float('inf')
        best_a = best_b = None
        for i_idx in range(len(nodes)):
            a = nodes[i_idx]
            for j_idx in range(i_idx + 1, len(nodes)):
                b = nodes[j_idx]
                criterion = (n_cur - 2) * dist_dict[(a, b)] - row_sum[a] - row_sum[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = a
                    best_b = b

        a, b = best_a, best_b
        d_ab = dist_dict[(a, b)]
        len_a = 0.5 * d_ab + (row_sum[a] - row_sum[b]) / (2.0 * (n_cur - 2))
        len_b = d_ab - len_a
        if len_a < 0:
            len_a = 0.0
        if len_b < 0:
            len_b = 0.0

        new_name = f"n{len(nodes)}_{a}_{b}"
        trees[new_name] = f"({trees[a]},{trees[b]})"

        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d_new_x = 0.5 * (dist_dict[(a, x)] + dist_dict[(b, x)] - d_ab)
            dist_dict[(new_name, x)] = d_new_x
            dist_dict[(x, new_name)] = d_new_x

        nodes = rest + [new_name]

    return "(" + ",".join(trees[x] for x in nodes) + ");"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # Compute distance matrix
    dist = _jc_dist_matrix(sequences, labels)

    if time.time() > deadline - 5.0:
        return "(" + ",".join(labels) + ");"

    # Quartet-informed distance adjustment
    dist_adj = _quartet_informed_dist(dist)

    if time.time() > deadline - 5.0:
        return "(" + ",".join(labels) + ");"

    # Run NJ on adjusted distances
    result = _nj_tree(dist_adj, labels)

    return result
