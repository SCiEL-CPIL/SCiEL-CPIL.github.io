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


def _embed(tree_edges, k, used_mask, rng, deadline):
    """
    Embed tree with k vertices into K20 using bitmask for used edges.
    Returns list of host vertices (length k) or None.
    """
    if k == 1:
        for h in range(20):
            return [h]
        return None

    if k == 2:
        for u in range(20):
            for v in range(u + 1, 20):
                mask = 1 << (u * 20 + v)
                mask2 = 1 << (v * 20 + u)
                if not (used_mask & mask) and not (used_mask & mask2):
                    return [u, v]
        return None

    # Build tree adjacency
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    deg = [len(adj[v]) for v in range(k)]
    
    # Try roots in order of decreasing degree (high-degree roots first)
    roots = sorted(range(k), key=lambda x: -deg[x])
    rng.shuffle(roots)

    for root in roots:
        if time.time() > deadline:
            return None

        # BFS/DFS to establish parent-child relationships
        visited = [False] * k
        parent = [-1] * k
        children = [[] for _ in range(k)]
        stack = [root]
        visited[root] = True
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = cur
                    children[cur].append(nb)
                    stack.append(nb)

        # Build assignment order: process vertices in order of decreasing "constrainedness"
        # A vertex is more constrained if it has many unassigned neighbors
        # We use a greedy approach: repeatedly pick the unassigned vertex with most assigned neighbors
        # (these are the most constrained)
        assigned_order = [root]
        assigned_set = {root}
        remaining = set(range(k)) - {root}
        
        while remaining:
            # Pick vertex with most assigned neighbors (most constrained)
            best_v = None
            best_count = -1
            for v in remaining:
                count = sum(1 for nb in adj[v] if nb in assigned_set)
                if count > best_count or (count == best_count and deg[v] > (deg[best_v] if best_v is not None else -1)):
                    best_count = count
                    best_v = v
            assigned_order.append(best_v)
            assigned_set.add(best_v)
            remaining.remove(best_v)

        # assignment[tv] = host vertex, or -1 if unassigned
        assign = [-1] * k
        used_host = [False] * 20

        def dfs(idx):
            if time.time() > deadline:
                return False

            if idx == len(assigned_order):
                return True

            tv = assigned_order[idx]

            if tv == root:
                # Try all available host vertices, prioritizing those with fewer used edges
                candidates = []
                for h in range(20):
                    if not used_host[h]:
                        # Count how many edges from h are already used
                        used_edges = 0
                        for nb in range(20):
                            if nb != h:
                                mask1 = 1 << (h * 20 + nb)
                                mask2 = 1 << (nb * 20 + h)
                                if used_mask & mask1 or used_mask & mask2:
                                    used_edges += 1
                        candidates.append((used_edges, h))
                candidates.sort()
                for _, h in candidates:
                    assign[tv] = h
                    used_host[h] = True
                    if dfs(idx + 1):
                        return True
                    used_host[h] = False
                    assign[tv] = -1
                return False
            else:
                # Find the parent of tv in the tree
                p = parent[tv]
                if p == -1:
                    # This shouldn't happen for non-root
                    return False
                ph = assign[p]
                if ph == -1:
                    # Parent not assigned yet - this shouldn't happen with our ordering
                    return False

                # Try all available host vertices adjacent to ph with unused edge
                candidates = []
                for h in range(20):
                    if h == ph or used_host[h]:
                        continue
                    mask1 = 1 << (ph * 20 + h)
                    mask2 = 1 << (h * 20 + ph)
                    if used_mask & mask1 or used_mask & mask2:
                        continue
                    # Score: prefer vertices that have fewer available edges for future constraints
                    # Count how many edges from h to unassigned neighbors of tv are still available
                    score = 0
                    for nb in adj[tv]:
                        if nb != p and assign[nb] == -1:
                            # Check if edge h-nb_host would be available
                            # We don't know nb_host yet, so just count potential constraints
                            score += 1
                    candidates.append((score, h))
                
                # Try most constrained first (higher score = more future constraints)
                candidates.sort(key=lambda x: -x[0])
                rng.shuffle(candidates)
                
                for _, h in candidates:
                    assign[tv] = h
                    used_host[h] = True
                    if dfs(idx + 1):
                        return True
                    used_host[h] = False
                    assign[tv] = -1
                return False

        if dfs(0):
            return [assign[i] for i in range(k)]

    return None


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin (5s before 150s kill)

    trees = _instance_trees(20, 611)

    best_result = [None] * 20
    best_value = 0

    for trial in range(500):
        if time.time() > deadline:
            break

        rng = Random(trial * 7919 + 31337)
        used_mask = 0
        result = [None] * 20
        total_edges = 0
        success = True

        # Embed in descending order of tree size, but with some randomization
        order = list(range(20, 0, -1))
        # For large trees, keep them first; for small ones, randomize slightly
        for i in range(len(order) - 1, 3, -1):
            j = rng.randint(max(4, i - 2), i)
            order[i], order[j] = order[j], order[i]

        for k in order:
            if time.time() > deadline:
                success = False
                break

            tree_edges = trees[k - 1]

            if not tree_edges:
                # T1: single vertex, pick any
                result[k - 1] = [rng.randrange(20)]
                continue

            embedding = _embed(tree_edges, k, used_mask, rng, deadline)

            if embedding is None:
                success = False
                break

            result[k - 1] = embedding
            # Mark edges as used
            for u, v in tree_edges:
                hu, hv = embedding[u], embedding[v]
                used_mask |= (1 << (hu * 20 + hv))
                used_mask |= (1 << (hv * 20 + hu))
            total_edges += len(tree_edges)

        if success and total_edges > best_value:
            best_value = total_edges
            best_result = result[:]
            if best_value == 190:
                break

    # Ensure correct types
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])

    return final
