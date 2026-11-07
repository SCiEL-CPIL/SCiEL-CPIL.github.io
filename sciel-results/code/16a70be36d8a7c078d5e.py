"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    trees = instance_trees(20, 611)
    num_trees = 20

    tree_data = []
    for i in range(num_trees):
        k = i + 1
        if k == 1:
            tree_data.append({'edges': [], 'adj': [[]], 'size': 1})
        elif k == 2:
            tree_data.append({'edges': [(0, 1)], 'adj': [[1], [0]], 'size': 2})
        else:
            edges = trees[i]
            adj = [[] for _ in range(k)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            tree_data.append({'edges': edges, 'adj': adj, 'size': k})

    # Precompute edge index for O(1) lookup: edge_idx[(u,v)] where u<v
    edge_index = {}
    idx = 0
    for u in range(20):
        for v in range(u + 1, 20):
            edge_index[(u, v)] = idx
            idx += 1
    total_edges = 190

    # Bitset for used edges
    def make_used_edges():
        return 0

    def set_edge(used, u, v):
        if u > v:
            u, v = v, u
        return used | (1 << edge_index[(u, v)])

    def is_edge_used(used, u, v):
        if u > v:
            u, v = v, u
        return (used >> edge_index[(u, v)]) & 1

    def embed_tree(tree_idx, used_edges_bitset, rng):
        k = tree_data[tree_idx]['size']
        if k == 1:
            for v in range(20):
                return [v]
            return None

        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    if not is_edge_used(used_edges_bitset, u, v):
                        return [u, v]
            return None

        adj = tree_data[tree_idx]['adj']
        edges = tree_data[tree_idx]['edges']
        degrees = [len(adj[i]) for i in range(k)]
        root = max(range(k), key=lambda i: (degrees[i], -i))

        # Precompute neighbor lists for each tree vertex
        # For forward checking, we need to know which unassigned neighbors exist

        best_result = None

        def backtrack(assigned, used_hosts, assigned_count):
            nonlocal best_result
            if assigned_count == k:
                # Verify all edges
                for u, v in edges:
                    hu = assigned[u]
                    hv = assigned[v]
                    if is_edge_used(used_edges_bitset, hu, hv):
                        return False
                best_result = assigned.copy()
                return True

            # MRV: find unassigned vertex with smallest valid domain
            unassigned = [v for v in range(k) if v not in assigned]
            best_v = -1
            min_dsize = 21
            best_domain = []

            for v in unassigned:
                valid_hosts = []
                for h in range(20):
                    if h in used_hosts:
                        continue
                    valid = True
                    for u in adj[v]:
                        if u in assigned:
                            hu = assigned[u]
                            if is_edge_used(used_edges_bitset, h, hu):
                                valid = False
                                break
                    if valid:
                        valid_hosts.append(h)

                dsize = len(valid_hosts)
                if dsize < min_dsize:
                    min_dsize = dsize
                    best_v = v
                    best_domain = valid_hosts
                    if dsize == 0:
                        break

            if best_v == -1 or min_dsize == 0:
                return False

            # Shuffle domain for diversity
            rng.shuffle(best_domain)

            for h in best_domain:
                assigned[best_v] = h
                used_hosts.add(h)

                # Forward check: ensure no unassigned neighbor has empty domain
                fc_ok = True
                for u in adj[best_v]:
                    if u not in assigned:
                        # Check if u has at least one valid host
                        has_valid = False
                        for h2 in range(20):
                            if h2 in used_hosts:
                                continue
                            ok = True
                            for w in adj[u]:
                                if w in assigned:
                                    hw = assigned[w]
                                    if is_edge_used(used_edges_bitset, h2, hw):
                                        ok = False
                                        break
                            if ok:
                                has_valid = True
                                break
                        if not has_valid:
                            fc_ok = False
                            break

                if fc_ok:
                    if backtrack(assigned, used_hosts, assigned_count + 1):
                        return True

                # Undo
                del assigned[best_v]
                used_hosts.discard(h)

            return False

        # Try each host for root
        root_hosts = list(range(20))
        rng.shuffle(root_hosts)

        for h in root_hosts:
            assigned = {root: h}
            used_hosts = {h}
            if backtrack(assigned, used_hosts, 1):
                mapping = [0] * k
                for tv, hv in best_result.items():
                    mapping[tv] = hv
                return mapping

        return None

    order = list(range(19, -1, -1))
    best_score = 0
    best_mappings = [None] * 20

    for restart in range(2000):
        if time.time() > deadline:
            break

        rng = Random(restart * 7919 + 31)
        used_edges = 0
        current_mappings = [None] * 20
        success = True

        for tree_idx in order:
            if time.time() > deadline:
                success = False
                break

            mapping = embed_tree(tree_idx, used_edges, rng)
            if mapping is None:
                success = False
                break
            current_mappings[tree_idx] = mapping
            # Update used edges
            for u, v in tree_data[tree_idx]['edges']:
                hu = mapping[u]
                hv = mapping[v]
                used_edges = set_edge(used_edges, hu, hv)

        if success:
            score = 0
            for i in range(20):
                if current_mappings[i] is not None:
                    score += len(tree_data[i]['edges'])

            if score > best_score:
                best_score = score
                best_mappings = [m[:] if m else None for m in current_mappings]

            if best_score == 190:
                break

    final_result = []
    for i in range(20):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])

    return final_result

# EVOLVE-BLOCK-END
