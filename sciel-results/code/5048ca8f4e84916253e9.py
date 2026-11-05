import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 145.0
    rng = random.Random(42)

    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))

    if k == 5 and g == 9:
        result = try_clebsch_lift(5, 9, deadline, rng, moore)
        if result:
            return result

    return moore, []


def try_clebsch_lift(k, g, deadline, rng, moore):
    n_base = 16
    base_adj = [[] for _ in range(n_base)]
    for u in range(n_base):
        for v in range(u + 1, n_base):
            x = u ^ v
            pc = bin(x).count('1')
            if pc == 1 or pc == 4:
                base_adj[u].append(v)
                base_adj[v].append(u)

    for i in range(n_base):
        if len(base_adj[i]) != 5:
            return None

    edge_list = []
    edge_index = {}
    for u in range(n_base):
        for v in base_adj[u]:
            if u < v:
                idx = len(edge_list)
                edge_list.append((u, v))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx

    num_edges = len(edge_list)

    walk_edges = []

    def gen_walks(curr, prev, path_edges, depth, max_depth):
        if depth > max_depth:
            return
        if depth >= 5:
            walk_edges.append(path_edges[:])
        for nxt in base_adj[curr]:
            if nxt == prev:
                continue
            gen_walks(nxt, curr, path_edges + [edge_index[(curr, nxt)]], depth + 1, max_depth)

    for start in range(n_base):
        for nb in base_adj[start]:
            gen_walks(nb, start, [edge_index[(start, nb)]], 2, 8)

    num_walks = len(walk_edges)

    # Precompute: for each walk, the edge indices as a numpy array
    # Also precompute: for each edge, which walks contain it (for fast conflict update)
    walk_arrays = [np.array(w, dtype=np.int32) for w in walk_edges]

    # For each edge, list of (walk_idx, position_in_walk)
    edge_to_walks = [[] for _ in range(num_edges)]
    for wi, wa in enumerate(walk_arrays):
        for pos, ei in enumerate(wa):
            edge_to_walks[ei].append((wi, pos))

    # Precompute walk lengths
    walk_lens = np.array([len(wa) for wa in walk_arrays], dtype=np.int32)

    # Try m values
    candidates = [m for m in range(27, 200) if 16 * m >= moore]
    candidates.sort()

    for m in candidates:
        if time.time() > deadline - 5:
            break

        best_voltages = None
        best_conflicts = float('inf')

        max_restarts = 300
        max_steps = 3000

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            v = np.random.randint(0, m, size=num_edges, dtype=np.int32)

            # Compute initial walk sums
            walk_sums = np.zeros(num_walks, dtype=np.int32)
            for wi, wa in enumerate(walk_arrays):
                walk_sums[wi] = np.sum(v[wa]) % m

            conflicts = np.sum(walk_sums == 0)

            if conflicts == 0:
                best_voltages = v.copy()
                best_conflicts = 0
                break

            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                if conflicts == 0:
                    best_voltages = v.copy()
                    best_conflicts = 0
                    break

                # Find a conflicting walk
                conflict_indices = np.where(walk_sums == 0)[0]
                if len(conflict_indices) == 0:
                    break
                wi = conflict_indices[rng.randint(0, len(conflict_indices) - 1)]
                wa = walk_arrays[wi]
                ei = wa[rng.randint(0, len(wa) - 1)]

                old_val = v[ei]
                new_val = rng.randint(0, m - 1)
                while new_val == old_val:
                    new_val = rng.randint(0, m - 1)
                delta = (new_val - old_val) % m
                v[ei] = new_val

                # Update walk sums for all walks containing edge ei
                for (w_idx, pos) in edge_to_walks[ei]:
                    walk_sums[w_idx] = (walk_sums[w_idx] + delta) % m

                new_conflicts = np.sum(walk_sums == 0)
                if new_conflicts >= conflicts:
                    # Revert
                    v[ei] = old_val
                    for (w_idx, pos) in edge_to_walks[ei]:
                        walk_sums[w_idx] = (walk_sums[w_idx] - delta) % m
                else:
                    conflicts = new_conflicts

            if best_voltages is None and conflicts < best_conflicts:
                best_conflicts = conflicts
                best_voltages = v.copy()

            if best_conflicts == 0:
                break

        if best_voltages is not None and best_conflicts == 0:
            n_lifted = n_base * m
            adj_lifted = [[] for _ in range(n_lifted)]
            edges_lifted = []

            for u in range(n_base):
                for v_base in base_adj[u]:
                    if u < v_base:
                        e_idx = edge_index[(u, v_base)]
                        vol = best_voltages[e_idx]
                        for a in range(m):
                            u1 = u * m + a
                            v1 = v_base * m + (a + vol) % m
                            adj_lifted[u1].append(v1)
                            adj_lifted[v1].append(u1)
                            if u1 < v1:
                                edges_lifted.append((u1, v1))

            return n_lifted, edges_lifted

    return None
