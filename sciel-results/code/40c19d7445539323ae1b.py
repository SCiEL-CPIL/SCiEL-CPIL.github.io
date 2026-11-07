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
    deadline = time.time() + 120
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
        # Find a good ordering: high degree first, then break ties by position
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

    def embed_tree(t_idx, available_edges, rng, max_time=5.0):
        if t_idx == 0:
            return [rng.randint(0, n - 1)]

        data = tree_data[t_idx]
        k = data['k']
        tree_adj = data['adj']
        ordered_tv = data['ordered']
        tree_edges = data['edges']
        start_time = time.time()

        # Build host adjacency from available edges
        host_adj = [set() for _ in range(n)]
        for (a, b) in available_edges:
            host_adj[a].add(b)
            host_adj[b].add(a)

        # Precompute for each tree vertex: its neighbors
        tv_neighbors = {v: tree_adj[v] for v in range(k)}

        host_map = {}
        used_hosts = set()
        assigned_count = 0
        tv_order = list(ordered_tv)

        # MRV-based dynamic ordering: always pick the unassigned vertex with fewest candidates
        def get_candidates(tv):
            """Get valid host candidates for tree vertex tv."""
            assigned_neighbors = [(nb, host_map[nb]) for nb in tv_neighbors[tv] if nb in host_map]

            if assigned_neighbors:
                candidates = set(host_adj[assigned_neighbors[0][1]])
                for _, hn in assigned_neighbors[1:]:
                    candidates &= host_adj[hn]
            else:
                candidates = set(range(n))

            candidates -= used_hosts
            return candidates

        def backtrack():
            nonlocal assigned_count
            if time.time() - start_time > max_time:
                return False
            if assigned_count == k:
                # Verify all edges are in available_edges
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available_edges:
                        return False
                return True

            # MRV: find unassigned vertex with fewest candidates
            best_tv = -1
            best_cands = None
            best_cand_count = float('inf')

            for tv in range(k):
                if tv in host_map:
                    continue
                cands = get_candidates(tv)
                if not cands:
                    return False
                if len(cands) < best_cand_count:
                    best_cand_count = len(cands)
                    best_tv = tv
                    best_cands = cands
                    if best_cand_count <= 1:
                        break

            if best_tv == -1:
                return False

            # Sort candidates: prefer hosts with fewer remaining edges (constrained first)
            cands_list = sorted(best_cands, key=lambda h: len(host_adj[h]))

            # Add slight randomness to explore different branches
            if rng.random() < 0.3 and len(cands_list) > 2:
                for i in range(len(cands_list) - 1, 0, -1):
                    j = rng.randint(0, i)
                    cands_list[i], cands_list[j] = cands_list[j], cands_list[i]

            for h in cands_list:
                host_map[best_tv] = h
                used_hosts.add(h)
                assigned_count += 1

                # Forward checking: verify all unassigned neighbors still have candidates
                valid = True
                for w in tv_neighbors[best_tv]:
                    if w in host_map:
                        continue
                    w_cands = get_candidates(w)
                    if not w_cands:
                        valid = False
                        break

                if valid:
                    if backtrack():
                        return True

                del host_map[best_tv]
                used_hosts.remove(h)
                assigned_count -= 1

            return False

        if backtrack():
            return [host_map[i] for i in range(k)]
        return None

    best_result = None
    best_edges = -1

    for attempt in range(500):
        if time.time() > deadline - 5:
            break

        rng = Random(attempt * 31 + 7)
        avail = set(all_edges)
        res = [None] * 20
        success = True

        for t_idx in tree_order:
            if time.time() > deadline - 5:
                success = False
                break
            k = t_idx + 1

            if k == 1:
                res[t_idx] = [rng.randint(0, n - 1)]
                continue

            embedded = None
            # Try multiple times with different random seeds for this tree
            for trial in range(8):
                if time.time() > deadline - 5:
                    success = False
                    break
                trial_rng = Random(attempt * 1000 + trial * 13 + 3)
                embedded = embed_tree(t_idx, avail, trial_rng, max_time=8.0)
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
            if total_edges > best_edges:
                best_edges = total_edges
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
