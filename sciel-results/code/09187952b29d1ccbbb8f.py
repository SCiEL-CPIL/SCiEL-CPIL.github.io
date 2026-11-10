import numpy as np
import time

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 155.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"

    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))

    seq_array = np.empty((n, L), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        seq_array[i] = [seq_map[c] for c in s]

    diff = (seq_array[:, None, :] != seq_array[None, :, :])
    hamming = diff.sum(axis=2)
    p = hamming / L
    D = np.zeros((n, n), dtype=np.float64)
    valid = (p > 0) & (p < 0.75)
    if np.any(valid):
        D[valid] = -0.75 * np.log(1.0 - (4.0 / 3.0) * p[valid])
    D[p >= 0.75] = 3.0
    np.fill_diagonal(D, 0.0)

    # --- Neighbor Joining ---
    def nj_build(dist):
        n_t = dist.shape[0]
        active = list(range(n_t))
        nodes = [i for i in range(n_t)]
        D_cur = dist.copy()

        while len(active) > 3:
            nn = len(active)
            r = D_cur.sum(axis=1)
            Q = (nn - 2) * D_cur - r[:, None] - r[None, :]
            np.fill_diagonal(Q, np.inf)
            idx = np.unravel_index(np.argmin(Q), Q.shape)
            i, j = idx
            node_i = active[i]
            node_j = active[j]
            new_node = (node_i, node_j)
            next_id = len(nodes)
            nodes.append(new_node)

            remaining = [k for k in range(nn) if k not in (i, j)]
            new_n = len(remaining) + 1
            new_D = np.zeros((new_n, new_n), dtype=np.float64)
            for a_idx, a in enumerate(remaining):
                for b_idx, b in enumerate(remaining):
                    new_D[a_idx, b_idx] = D_cur[a, b]
                dist_new = 0.5 * (D_cur[i, a] + D_cur[j, a] - D_cur[i, j])
                new_D[a_idx, new_n - 1] = max(dist_new, 0.0)
                new_D[new_n - 1, a_idx] = max(dist_new, 0.0)
            active = [active[k] for k in remaining] + [next_id]
            D_cur = new_D

        a, b, c = active
        return (nodes[a], (nodes[b], nodes[c]))

    def build_newick(node, label_map):
        if isinstance(node, int):
            return label_map[node]
        left, right = node
        return "(" + build_newick(left, label_map) + "," + build_newick(right, label_map) + ")"

    label_map = list(range(n))
    tree = nj_build(D)

    return build_newick(tree, label_map) + ";"
