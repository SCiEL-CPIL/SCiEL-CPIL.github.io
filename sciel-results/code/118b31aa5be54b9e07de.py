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
    deadline = start_time + 115

    trees = instance_trees(20, 611)

    tree_edges = []
    for t in trees:
        e = set()
        for u, v in t:
            e.add((min(u, v), max(u, v)))
        tree_edges.append(e)

    N = 20
    all_host_edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_host_edges.add((i, j))

    tree_degrees = []
    tree_adj = []
    for k_idx in range(20):
        k = k_idx + 1
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for u, v in trees[k_idx]:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        tree_degrees.append(deg)
        tree_adj.append(adj)

    def time_up():
        return time.time() > deadline

    def embed_tree(tree_idx, available_edges, rng, timeout_sec=5.0):
        k = tree_idx + 1
        if k == 1:
            return [0]
        if k == 2:
            if not available_edges:
                return None
            e = next(iter(available_edges))
            return [e[0], e[1]]

        deg = tree_degrees[tree_idx]
        adj = tree_adj[tree_idx]

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

        node_timeout = start_time + timeout_sec

        def dfs(idx):
            if time.time() > node_timeout:
                return False
            if idx == k:
                return True

            v = order[idx]
            v_deg = deg[v]

            assigned_neighbors = []
            for u in adj[v]:
                if pos_in_order[u] < idx:
                    assigned_neighbors.append((u, assigned[pos_in_order[u]]))

            candidates = []
            for h in range(N):
                if h in used_hosts:
                    continue
                valid = True
                for u, hu in assigned_neighbors:
                    edge = (min(h, hu), max(h, hu))
                    if edge not in available_edges:
                        valid = False
                        break
                if not valid:
                    continue
                if rem_deg[h] < v_deg:
                    continue

                # Forward checking: for each unassigned neighbor w of v,
                # after assigning h to v, w must still be feasible
                fc_ok = True
                for w in adj[v]:
                    if pos_in_order[w] >= idx:
                        # w will have 1 edge to h (the v-w tree edge)
                        # w needs deg[w] total edges from distinct hosts
                        # Count w's already-assigned neighbors (excluding v which is now h)
                        assigned_w_count = 0
                        for x in adj[w]:
                            if pos_in_order[x] < idx:
                                assigned_w_count += 1
                        # w needs deg[w] - assigned_w_count - 1 more edges to unassigned hosts
                        # (minus 1 for the edge to h)
                        # The host h will have rem_deg[h] - 1 edges left after using one for w
                        # But more importantly, w needs enough distinct available hosts
                        # Simple check: there must be enough unassigned hosts with available edges
                        remaining_hosts = N - len(used_hosts) - 1  # minus h
                        needed = deg[w] - assigned_w_count - 1
                        if needed > remaining_hosts:
                            fc_ok = False
                            break
                        # Also check that h has enough remaining degree after all its edges
                        # h uses v_deg edges total, so rem_deg[h] - v_deg >= 0
                        # (already checked rem_deg[h] >= v_deg)
                        # Check that after h uses 1 edge to w's future host, 
                        # w's future host still has capacity
                        # This is complex; use a lighter check:
                        # For each neighbor x of w that is unassigned, x needs deg[x] edges
                        # We just need to ensure the graph isn't too constrained
                        pass

                if fc_ok:
                    candidates.append(h)

            if not candidates:
                return False

            # Sort by rem_deg ascending (most constrained first)
            candidates.sort(key=lambda h: rem_deg[h])
            # Add some randomness to escape local minima
            if len(candidates) > 1:
                # Partial shuffle: swap a few random elements
                for _ in range(min(3, len(candidates) - 1)):
                    i = rng.randint(0, len(candidates) - 2)
                    j = rng.randint(i + 1, len(candidates) - 1)
                    candidates[i], candidates[j] = candidates[j], candidates[i]

            for h in candidates:
                assigned[pos_in_order[v]] = h
                used_hosts.add(h)

                old_rem_h = rem_deg[h]
                rem_deg[h] -= len(assigned_neighbors)
                old_rems = []
                for u, hu in assigned_neighbors:
                    old_rems.append(rem_deg[hu])
                    rem_deg[hu] -= 1

                if dfs(idx + 1):
                    return True

                assigned[pos_in_order[v]] = -1
                used_hosts.discard(h)
                rem_deg[h] = old_rem_h
                for i, (u, hu) in enumerate(assigned_neighbors):
                    rem_deg[hu] = old_rems[i]

            return False

        if dfs(0):
            mapping = [0] * k
            for i, v in enumerate(order):
                mapping[v] = assigned[i]
            return mapping
        return None

    def get_used_edges(mapping, tree_idx):
        edges = set()
        for u, v in trees[tree_idx]:
            hu, hv = mapping[u], mapping[v]
            edges.add((min(hu, hv), max(hu, hv)))
        return edges

    def try_packing(order, rng_master, max_attempts_per_tree=5):
        mappings = [None] * 20
        used_edges = set()
        available = set(all_host_edges)
        value = 0

        for tree_idx in order:
            if time_up():
                break
            k = tree_idx + 1
            if k == 1:
                mappings[tree_idx] = [0]
                continue

            found = False
            for attempt in range(max_attempts_per_tree):
                if time_up():
                    break
                rng = Random(rng_master * 100000 + tree_idx * 1000 + attempt * 7)
                m = embed_tree(tree_idx, available, rng, timeout_sec=3.0)
                if m is not None:
                    mappings[tree_idx] = m
                    used_edges |= get_used_edges(m, tree_idx)
                    available -= get_used_edges(m, tree_idx)
                    value += len(tree_edges[tree_idx])
                    found = True
                    break

            if not found:
                mappings[tree_idx] = None

        return mappings, value

    max_deg = [max(tree_degrees[i]) if tree_degrees[i] else 0 for i in range(20)]
    edge_counts = [len(tree_edges[i]) for i in range(20)]

    order1 = list(range(19, -1, -1))
    order2 = list(range(20))
    order3 = sorted(range(20), key=lambda i: -max_deg[i])
    order4 = sorted(range(20), key=lambda i: -edge_counts[i])
    # Mixed: large trees with high degree first
    order5 = sorted(range(20), key=lambda i: (-(i + 1), -max_deg[i]))

    best_mappings = [None] * 20
    best_value = 0

    strategies = [
        (order1, 1), (order1, 42), (order1, 123), (order1, 7),
        (order4, 1), (order4, 42), (order4, 123), (order4, 7),
        (order3, 1), (order3, 42), (order3, 123), (order3, 7),
        (order5, 1), (order5, 42), (order5, 123), (order5, 7),
        (order2, 1), (order2, 42),
    ]

    for order, rng_master in strategies:
        if time_up():
            break
        mappings, value = try_packing(order, rng_master, max_attempts_per_tree=4)
        if value > best_value:
            best_value = value
            best_mappings = list(mappings)
        if best_value >= 190:
            break

    if best_value < 190 and not time_up():
        missing = [i for i in range(20) if best_mappings[i] is None]
        if missing:
            used_edges = set()
            for i in range(20):
                if best_mappings[i] is not None:
                    used_edges |= get_used_edges(best_mappings[i], i)
            available = all_host_edges - used_edges

            missing_sorted = sorted(missing, key=lambda i: -(i + 1))
            for tree_idx in missing_sorted:
                if time_up():
                    break
                k = tree_idx + 1
                if k == 1:
                    best_mappings[tree_idx] = [0]
                    continue
                found = False
                for attempt in range(30):
                    if time_up():
                        break
                    rng = Random(99999 + tree_idx * 1000 + attempt)
                    m = embed_tree(tree_idx, available, rng, timeout_sec=2.0)
                    if m is not None:
                        best_mappings[tree_idx] = m
                        used_edges |= get_used_edges(m, tree_idx)
                        available -= get_used_edges(m, tree_idx)
                        best_value += len(tree_edges[tree_idx])
                        found = True
                        break

                if not found and not time_up():
                    placed = [i for i in range(20) if best_mappings[i] is not None and i != tree_idx]
                    if placed:
                        remove_idx = max(placed, key=lambda i: (i + 1))
                        removed_edges = get_used_edges(best_mappings[remove_idx], remove_idx)
                        used_edges -= removed_edges
                        available = all_host_edges - used_edges
                        best_mappings[remove_idx] = None
                        best_value -= len(tree_edges[remove_idx])

                        for t_idx in [tree_idx, remove_idx]:
                            if time_up():
                                break
                            tk = t_idx + 1
                            if tk == 1:
                                best_mappings[t_idx] = [0]
                                continue
                            for attempt in range(15):
                                if time_up():
                                    break
                                rng = Random(77777 + t_idx * 500 + attempt)
                                m = embed_tree(t_idx, available, rng, timeout_sec=2.0)
                                if m is not None:
                                    best_mappings[t_idx] = m
                                    used_edges |= get_used_edges(m, t_idx)
                                    available -= get_used_edges(m, t_idx)
                                    best_value += len(tree_edges[t_idx])
                                    break

    result = []
    for i in range(20):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])

    return result
