"""Rewritten packing program with stronger search heuristics."""
# EVOLVE-BLOCK-START
import time
from random import Random


def _instance_trees(n=20, seed=611):
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


def _embed_tree(tree_edges, k, used_edges, rng, max_attempts=50, timeout=8.0):
    """Embed tree into K20 using unused edges via randomized DFS with backtracking."""
    if k == 1:
        return [rng.randint(0, 19)]
    if k == 2:
        for u in range(20):
            for v in range(u + 1, 20):
                if (u, v) not in used_edges:
                    return [u, v]
        return None

    # Build adjacency
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Degree ordering: sort vertices by degree descending for better search
    # but we need to maintain tree structure, so use a leaf-removal ordering
    # Actually, let's use a BFS/DFS from a random root each attempt

    start = time.time()
    for attempt in range(max_attempts):
        if time.time() - start > timeout:
            break

        root = rng.randint(0, k - 1)

        # Build DFS order from root
        visited = [False] * k
        order = []
        parent = [-1] * k
        stack = [root]
        visited[root] = True
        while stack:
            cur = stack.pop()
            order.append(cur)
            # Shuffle neighbors for randomness
            nbs = adj[cur][:]
            rng.shuffle(nbs)
            for nb in nbs:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = cur
                    stack.append(nb)

        assignment = [-1] * k
        available = [True] * 20

        def dfs(idx):
            if time.time() - start > timeout:
                return False
            if idx == len(order):
                return True
            tv = order[idx]
            if tv == root:
                hosts = [h for h in range(20) if available[h]]
                rng.shuffle(hosts)
                for h in hosts:
                    assignment[tv] = h
                    available[h] = False
                    if dfs(idx + 1):
                        return True
                    available[h] = True
                    assignment[tv] = -1
                return False
            else:
                p = parent[tv]
                ph = assignment[p]
                if ph == -1:
                    return False
                hosts = []
                for h in range(20):
                    if available[h] and h != ph:
                        if (ph, h) not in used_edges:
                            hosts.append(h)
                rng.shuffle(hosts)
                for h in hosts:
                    assignment[tv] = h
                    available[h] = False
                    if dfs(idx + 1):
                        return True
                    available[h] = True
                    assignment[tv] = -1
                return False

        if dfs(0):
            # Verify all edges
            for u, v in tree_edges:
                hu, hv = assignment[u], assignment[v]
                if (hu, hv) in used_edges or (hv, hu) in used_edges:
                    return None
            return [assignment[i] for i in range(k)]

    return None


def _embed_tree_greedy(tree_edges, k, used_edges, rng, max_attempts=30):
    """Greedy embedding with leaf-stripping order."""
    if k == 1:
        return [rng.randint(0, 19)]
    if k == 2:
        for u in range(20):
            for v in range(u + 1, 20):
                if (u, v) not in used_edges:
                    return [u, v]
        return None

    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    start = time.time()
    for attempt in range(max_attempts):
        if time.time() - start > 8.0:
            break

        # Leaf-stripping: repeatedly pick a leaf, map it, remove it
        remaining = list(range(k))
        rem_adj = [set(adj[i]) for i in range(k)]
        assignment = [-1] * k

        # Pick random root first
        root = rng.choice(remaining)
        assignment[root] = -1  # will be assigned below

        # Order: root first, then leaves in removal order
        order = [root]
        temp_rem = remaining[:]
        temp_adj = [set(adj[i]) for i in range(k)]
        temp_adj[root] = set()  # remove root's adjacency for leaf finding

        while temp_rem:
            # Find a leaf in remaining subgraph
            leaves = [v for v in temp_rem if len([nb for nb in temp_adj[v] if nb in temp_rem]) <= 1]
            if not leaves:
                leaves = temp_rem[:]
            leaf = rng.choice(leaves)
            order.append(leaf)
            temp_rem.remove(leaf)
            for nb in list(temp_adj[leaf]):
                if nb in temp_rem:
                    temp_adj[nb].discard(leaf)
            temp_adj[leaf] = set()

        # Now assign in order
        available = [True] * 20

        def assign_in_order(idx):
            if idx == len(order):
                return True
            tv = order[idx]
            if idx == 0:
                hosts = [h for h in range(20) if available[h]]
                rng.shuffle(hosts)
                for h in hosts:
                    assignment[tv] = h
                    available[h] = False
                    if assign_in_order(idx + 1):
                        return True
                    available[h] = True
                    assignment[tv] = -1
                return False
            else:
                # Find parent in remaining
                p = None
                for nb in adj[tv]:
                    if assignment[nb] != -1:
                        p = nb
                        break
                if p is None:
                    return False
                ph = assignment[p]
                hosts = [h for h in range(20) if available[h] and h != ph and (ph, h) not in used_edges]
                rng.shuffle(hosts)
                for h in hosts:
                    assignment[tv] = h
                    available[h] = False
                    if assign_in_order(idx + 1):
                        return True
                    available[h] = True
                    assignment[tv] = -1
                return False

        if assign_in_order(0):
            for u, v in tree_edges:
                hu, hv = assignment[u], assignment[v]
                if (hu, hv) in used_edges:
                    return None
            return [assignment[i] for i in range(k)]

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140

    trees = _instance_trees(20, 611)

    best_result = [None] * 19 + [list(range(20))]
    best_value = 19

    # Precompute tree edge counts
    tree_sizes = [len(t) for t in trees]

    for trial in range(800):
        if time.time() > deadline:
            break

        rng = Random(trial * 104729 + 987654321)
        used_edges = set()
        result = [None] * 20
        total_edges = 0
        success = True

        # Choose embedding order: try multiple strategies
        if trial % 4 == 0:
            order = list(range(1, 21))  # ascending
        elif trial % 4 == 1:
            order = list(range(20, 0, -1))  # descending
        elif trial % 4 == 2:
            # Larger first, but T1-T2 last (trivial)
            order = list(range(20, 2, -1)) + [1, 2]
        else:
            # Random shuffle
            order = list(range(1, 21))
            rng.shuffle(order)

        for k in order:
            if time.time() > deadline:
                success = False
                break

            tree_edges = trees[k - 1]
            if not tree_edges:
                avail = [v for v in range(20)]
                result[k - 1] = [rng.choice(avail)]
                continue

            # Try greedy first, then backtrack
            embedding = _embed_tree_greedy(tree_edges, k, used_edges, rng, max_attempts=20)
            if embedding is None:
                embedding = _embed_tree(tree_edges, k, used_edges, rng, max_attempts=80, timeout=6.0)

            if embedding is None:
                success = False
                break

            result[k - 1] = embedding
            for u, v in tree_edges:
                hu, hv = embedding[u], embedding[v]
                used_edges.add((hu, hv))
                used_edges.add((hv, hu))
            total_edges += len(tree_edges)

        if success and total_edges > best_value:
            best_value = total_edges
            best_result = result[:]
            if best_value == 190:
                break

    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])

    return final
# EVOLVE-BLOCK-END
