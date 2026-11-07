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
    deadline = start_time + 140

    trees = instance_trees(20, 611)
    N = 20

    # Precompute tree info
    tree_edges = []
    tree_deg = []
    tree_adj = []
    for k_idx in range(20):
        k = k_idx + 1
        es = set()
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for u, v in trees[k_idx]:
            es.add((min(u, v), max(u, v)))
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        tree_edges.append(es)
        tree_deg.append(deg)
        tree_adj.append(adj)

    all_host_edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_host_edges.add((i, j))

    def time_up():
        return time.time() > deadline

    def embed_tree(k_idx, available_edges, rng):
        """Embed tree k_idx into K20 using only edges in available_edges."""
        k = k_idx + 1
        if k == 1:
            return [0]
        if k == 2:
            for e in available_edges:
                return [e[0], e[1]]
            return None

        deg = tree_deg[k_idx]
        adj = tree_adj[k_idx]

        # Build degree-sorted order for DFS
        order = sorted(range(k), key=lambda x: -deg[x])
        pos_in_order = [0] * k
        for i, v in enumerate(order):
            pos_in_order[v] = i

        assigned = [-1] * k
        used_hosts = set()
        rem_deg = [0] * N
        for e in available_edges:
            rem_deg[e[0]] += 1
            rem_deg[e[1]] += 1

        # Precompute for each tree vertex: which tree neighbors come before it in order
        before_neighbors = [[] for _ in range(k)]  # before_neighbors[v] = list of (neighbor, pos)
        for v in range(k):
            for u in adj[v]:
                if pos_in_order[u] < pos_in_order[v]:
                    before_neighbors[v].append((u, pos_in_order[u]))

        def dfs(idx):
            if time_up():
                return False
            if idx == k:
                return True

            v = order[idx]
            v_deg = deg[v]
            bneigh = before_neighbors[v]

            # Collect candidate host vertices
            candidates = []
            for h in range(N):
                if h in used_hosts:
                    continue
                # Check edges to already-assigned neighbors
                ok = True
                for u, pu in bneigh:
                    hu = assigned[pu]
                    edge = (min(h, hu), max(h, hu))
                    if edge not in available_edges:
                        ok = False
                        break
                if not ok:
                    continue
                # Degree feasibility: h must have enough remaining degree
                if rem_deg[h] < v_deg:
                    continue
                candidates.append(h)

            if not candidates:
                return False

            # LCV: sort by remaining degree (most constrained first), add randomness
            candidates.sort(key=lambda h: (rem_deg[h], rng.random()))

            for h in candidates:
                assigned[pos_in_order[v]] = h
                used_hosts.add(h)

                # Update rem_deg
                old_rem_h = rem_deg[h]
                rem_deg[h] -= len(bneigh)
                old_rems = []
                for u, pu in bneigh:
                    hu = assigned[pu]
                    old_rems.append(rem_deg[hu])
                    rem_deg[hu] -= 1

                if dfs(idx + 1):
                    return True

                # Undo
                assigned[pos_in_order[v]] = -1
                used_hosts.discard(h)
                rem_deg[h] = old_rem_h
                for i, (u, pu) in enumerate(bneigh):
                    hu = assigned[pu]
                    rem_deg[hu] = old_rems[i]

            return False

        if dfs(0):
            mapping = [0] * k
            for i, v in enumerate(order):
                mapping[v] = assigned[i]
            return mapping
        return None

    def get_used_edges(mapping, k_idx):
        edges = set()
        for u, v in trees[k_idx]:
            hu, hv = mapping[u], mapping[v]
            edges.add((min(hu, hv), max(hu, hv)))
        return edges

    def try_packing(order, seed):
        mappings = [None] * 20
        used_edges = set()
        available = set(all_host_edges)
        value = 0
        rng = Random(seed)

        for k_idx in order:
            if time_up():
                break
            k = k_idx + 1
            if k == 1:
                mappings[k_idx] = [0]
                continue

            found = False
            for attempt in range(50):
                if time_up():
                    break
                arng = Random(seed * 100000 + k_idx * 1000 + attempt)
                m = embed_tree(k_idx, available, arng)
                if m is not None:
                    mappings[k_idx] = m
                    used_edges |= get_used_edges(m, k_idx)
                    available = all_host_edges - used_edges
                    value += len(tree_edges[k_idx])
                    found = True
                    break

            if not found:
                mappings[k_idx] = None

        return mappings, value

    # Multiple packing orders
    orders = [
        list(range(19, -1, -1)),  # descending size
        list(range(20)),          # ascending size
        sorted(range(20), key=lambda i: -max(tree_deg[i]) if tree_deg[i] else 0),  # max degree
        sorted(range(20), key=lambda i: -len(tree_edges[i])),  # edge count
    ]

    best_mappings = [None] * 20
    best_value = 0

    seeds = [1, 7, 42, 123, 999, 31415, 271828, 161803, 555, 888]

    for oi, order in enumerate(orders):
        for si, seed in enumerate(seeds):
            if time_up():
                break
            mappings, value = try_packing(order, seed)
            if value > best_value:
                best_value = value
                best_mappings = mappings[:]
            if best_value >= 190:
                break
        if best_value >= 190:
            break

    # If not full, try to fill gaps
    if best_value < 190 and not time_up():
        used_edges = set()
        for i in range(20):
            if best_mappings[i] is not None:
                used_edges |= get_used_edges(best_mappings[i], i)
        available = all_host_edges - used_edges

        missing = sorted([i for i in range(20) if best_mappings[i] is None], key=lambda i: -(i + 1))

        for k_idx in missing:
            if time_up():
                break
            k = k_idx + 1
            if k == 1:
                best_mappings[k_idx] = [0]
                continue
            found = False
            for attempt in range(200):
                if time_up():
                    break
                arng = Random(999999 + k_idx * 1000 + attempt)
                m = embed_tree(k_idx, available, arng)
                if m is not None:
                    best_mappings[k_idx] = m
                    used_edges |= get_used_edges(m, k_idx)
                    available = all_host_edges - used_edges
                    best_value += len(tree_edges[k_idx])
                    found = True
                    break

    # Final: ensure valid output format
    result = []
    for i in range(20):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])

    return result
