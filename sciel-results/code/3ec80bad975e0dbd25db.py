import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 130
    trees = instance_trees(20, 611)
    n = 20

    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Precompute tree data
    tree_data = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k == 1:
            tree_data.append(None)
            continue
        tree_adj = {v: [] for v in range(k)}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        ordered_tv = sorted(range(k), key=lambda v: (-len(tree_adj[v]), v))
        tree_data.append({
            'edges': tree_edges,
            'adj': tree_adj,
            'ordered': ordered_tv,
            'k': k
        })

    # Largest trees first
    tree_order = list(range(20))
    tree_order.sort(key=lambda idx: (-(idx + 1), -idx))

    def embed_tree(t_idx, available_edges, rng):
        if t_idx == 0:
            return [rng.randint(0, n - 1)]

        data = tree_data[t_idx]
        k = data['k']
        tree_adj = data['adj']
        ordered_tv = data['ordered']
        tree_edges = data['edges']

        # Build host adjacency from available edges
        host_adj = [set() for _ in range(n)]
        for (a, b) in available_edges:
            host_adj[a].add(b)
            host_adj[b].add(a)

        host_map = {}
        used_hosts = set()
        tv_order = list(ordered_tv)
        tv_pos = {v: i for i, v in enumerate(tv_order)}

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == len(tv_order):
                # Verify all edges
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available_edges:
                        return False
                return True

            tv = tv_order[pos]
            tv_neighbors = tree_adj[tv]

            # Determine candidates for tv
            assigned_neighbors = [(nb, host_map[nb]) for nb in tv_neighbors if nb in host_map]

            if assigned_neighbors:
                candidates = set(host_adj[assigned_neighbors[0][1]])
                for _, hn in assigned_neighbors[1:]:
                    candidates &= host_adj[hn]
            else:
                candidates = set(range(n))

            candidates -= used_hosts

            if not candidates:
                return False

            # Sort by degree in host graph (descending) for better pruning
            candidates_list = sorted(candidates, key=lambda h: -len(host_adj[h]))

            # Add randomness to explore different branches
            if rng.random() < 0.4 and len(candidates_list) > 2:
                # Shuffle a random subset
                for i in range(len(candidates_list) - 1, 0, -1):
                    j = rng.randint(0, i)
                    candidates_list[i], candidates_list[j] = candidates_list[j], candidates_list[i]

            for h in candidates_list:
                host_map[tv] = h
                used_hosts.add(h)

                # Forward checking: verify unassigned neighbors still have candidates
                valid = True
                for w in tv_neighbors:
                    if w in host_map:
                        continue
                    # w is unassigned, adjacent to tv (now at h)
                    # w's other assigned neighbors:
                    w_assigned = [host_map[nb] for nb in tree_adj[w] if nb in host_map and nb != tv]
                    # Candidates for w: in host_adj[h] AND in host_adj[x] for all x in w_assigned, minus used
                    inter = set(host_adj[h])
                    for x in w_assigned:
                        inter &= host_adj[x]
                    inter -= used_hosts
                    if not inter:
                        valid = False
                        break

                if valid:
                    if backtrack(pos + 1):
                        return True

                del host_map[tv]
                used_hosts.remove(h)

            return False

        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    best_result = None

    for attempt in range(200):
        if time.time() > deadline:
            break

        rng = Random(attempt * 31 + 7)
        avail = set(all_edges)
        res = [None] * 20
        success = True

        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            k = t_idx + 1

            if k == 1:
                res[t_idx] = [rng.randint(0, n - 1)]
                continue

            embedded = None
            for trial in range(12):
                if time.time() > deadline:
                    success = False
                    break
                embedded = embed_tree(t_idx, avail, rng)
                if embedded is not None:
                    break

            if not success:
                break
            if embedded is None:
                success = False
                break

            res[t_idx] = embedded
            for u, v in trees[t_idx]:
                hu, hv = embedded[u], embedded[v]
                edge = (min(hu, hv), max(hu, hv))
                avail.discard(edge)

        if success:
            # Count embedded edges
            total_edges = sum(len(trees[i]) for i in range(20) if res[i] is not None)
            if best_result is None or total_edges > sum(len(trees[i]) for i in range(20) if best_result[i] is not None):
                best_result = res
                if total_edges == 190:  # All edges used
                    break

    if best_result is None:
        return [None] * 19 + [list(range(20))]

    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final
