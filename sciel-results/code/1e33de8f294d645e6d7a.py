import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 142.0

    def elapsed():
        return time.time() - start_time

    def try_build(n, seed, max_iters=100000):
        if elapsed() > DEADLINE - 3:
            return None

        rng = np.random.default_rng(seed)
        min_deg_needed = n - 39

        adj = np.ones((n, n), dtype=np.int8)
        np.fill_diagonal(adj, 0)
        deg = np.sum(adj, axis=1).copy()

        for it in range(max_iters):
            if elapsed() > DEADLINE - 3:
                return None

            common = adj @ adj.T
            non_adj = (adj == 0)
            np.fill_diagonal(non_adj, False)
            c4_mask = non_adj & (common >= 2)

            if not c4_mask.any():
                if deg.min() >= min_deg_needed:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append([int(i), int(j)])
                    return {'n': n, 'edges': edges}
                else:
                    return None

            pairs = np.argwhere(c4_mask)
            idx = rng.integers(0, len(pairs))
            u, v = int(pairs[idx][0]), int(pairs[idx][1])

            cn_mask = (adj[u] & adj[v]).copy()
            cn_mask[u] = 0
            cn_mask[v] = 0
            cn = np.where(cn_mask)[0]

            if len(cn) < 2:
                continue

            w = int(cn[rng.integers(0, len(cn))])
            mask2 = cn != w
            x = int(cn[mask2][rng.integers(0, mask2.sum())])

            c4_edges = [(u, w), (w, v), (v, x), (x, u)]

            valid = [(a, b) for a, b in c4_edges if deg[a] > min_deg_needed and deg[b] > min_deg_needed]
            if not valid:
                valid = c4_edges

            best = max(valid, key=lambda ab: common[ab[0], ab[1]])
            a, b = best
            adj[a, b] = 0
            adj[b, a] = 0
            deg[a] -= 1
            deg[b] -= 1

        return None

    for n in [47, 46]:
        if elapsed() > DEADLINE - 10:
            break
        for seed in range(20000):
            if elapsed() > DEADLINE - 3:
                break
            result = try_build(n, seed)
            if result is not None:
                return result

    return {'n': 39, 'edges': []}
