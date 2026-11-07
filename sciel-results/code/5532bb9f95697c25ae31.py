import time
from random import Random
from collections import deque

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
    deadline = start_time + 115  # 115s limit to be safe under 150s

    trees = instance_trees(20, 611)

    # Precompute tree edge sets (undirected)
    tree_edges = []
    for t in trees:
        e = set()
        for u, v in t:
            e.add((min(u, v), max(u, v)))
        tree_edges.append(e)

    # Host graph K20
    N = 20
    all_host_edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_host_edges.add((i, j))

    # Precompute tree degrees and adjacency
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

    def embed_tree_greedy(tree_idx, available_edges, rng):
        """
        Greedy leaf-pruning embedding: repeatedly pick a leaf of the remaining
        tree and assign it to a host vertex that has an available edge to its
        already-assigned parent. This is fast and often succeeds.
        Returns mapping list or None.
        """
        k = tree_idx + 1
        if k == 1:
            return [0]
        if k == 2:
            for e in available_edges:
                return [e[0], e[1]]
            return None

        deg = tree_degrees[tree_idx]
        adj = tree_adj[tree_idx]

        # Build a spanning tree rooted at highest-degree vertex
        root = max(range(k), key=lambda x: deg[x])
        
        # BFS/DFS to get parent pointers
        parent = [-1] * k
        order = []  # BFS order
        visited = [False] * k
        queue = deque([root])
        visited[root] = True
        while queue:
            v = queue.popleft()
            order.append(v)
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    parent[u] = v
                    queue.append(u)

        # Process in reverse BFS order (leaves first)
        # assigned[v] = host vertex for tree vertex v
        assigned = [-1] * k
        used_hosts = set()

        # Assign root first
        # Pick a host with highest degree in available graph
        host_deg = [0] * N
        for e in available_edges:
            host_deg[e[0]] += 1
            host_deg[e[1]] += 1
        
        # Try several random roots for the host assignment
        for attempt in range(50):
            if time_up():
                return None
            
            assigned = [-1] * k
            used_hosts = set()
            
            # Choose root host
            # Prefer hosts with high degree
            candidates = sorted(range(N), key=lambda h: -host_deg[h])
            rng.shuffle(candidates[:10])
            root_host = candidates[0]
            
            assigned[root] = root_host
            used_hosts.add(root_host)
            
            success = True
            # Process in reverse BFS order (children before parents, but we need parent assigned first)
            # Actually, process in BFS order but assign children to parents
            # Let's redo: process in reverse order (leaves first), but we need parent assigned
            # So process in BFS order: root, then its children, etc.
            
            for v in order:
                if v == root:
                    continue
                p = parent[v]
                ph = assigned[p]
                if ph == -1:
                    success = False
                    break
                
                # Find a host h such that (h, ph) in available_edges and h not in used_hosts
                found = False
                # Get neighbors of ph in available graph
                neighbors = [u for u, v2 in available_edges if u == ph or v2 == ph]
                # Actually, let's be more efficient
                neigh_ph = []
                for e in available_edges:
                    if e[0] == ph:
                        neigh_ph.append(e[1])
                    elif e[1] == ph:
                        neigh_ph.append(e[0])
                
                # Shuffle neighbors for diversity
                rng.shuffle(neigh_ph)
                
                for h in neigh_ph:
                    if h not in used_hosts:
                        assigned[v] = h
                        used_hosts.add(h)
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if success:
                return assigned
        
        return None

    def embed_tree_backtrack(tree_idx, available_edges, rng):
        """
        Backtracking embedding with better heuristics.
        """
        k = tree_idx + 1
        if k == 1:
            return [0]
        if k == 2:
            for e in available_edges:
                return [e[0], e[1]]
            return None

        deg = tree_degrees[tree_idx]
        adj = tree_adj[tree_idx]

        # Order tree vertices by degree descending
        order = sorted(range(k), key=lambda x: -deg[x])
        pos_in_order = [0] * k
        for i, v in enumerate(order):
            pos_in_order[v] = i

        assigned = [-1] * k
        used_hosts = set()

        # Precompute available neighbors for each host
        host_neighbors = [[] for _ in range(N)]
        for e in available_edges:
            host_neighbors[e[0]].append(e[1])
            host_neighbors[e[1]].append(e[0])

        # Track which edges are used (for pruning)
        # rem_deg[h] = number of available edges incident to h
        rem_deg = [len(host_neighbors[h]) for h in range(N)]

        nodes_visited = [0]

        def dfs(idx):
            if time_up():
                return False
            nodes_visited[0] += 1
            if nodes_visited[0] > 500000:
                return False
            if idx == k:
                return True

            v = order[idx]
            v_deg = deg[v]

            # Get already-assigned neighbors of v
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
                candidates.append(h)

            if not candidates:
                return False

            # MRV: prefer most constrained
            candidates.sort(key=lambda h: rem_deg[h])
            rng.shuffle(candidates)

            for h in candidates:
                assigned[pos_in_order[v]] = h
                used_hosts.add(h)
                old_rem = rem_deg[h]
                rem_deg[h] -= 1

                if dfs(idx + 1):
                    return True

                assigned[pos_in_order[v]] = -1
                used_hosts.discard(h)
                rem_deg[h] = old_rem

            return False

        if dfs(0):
            mapping = [0] * k
            for i, v in enumerate(order):
                mapping[v] = assigned[i]
            return mapping
        return None

    def get_used_edges(mapping, tree_idx):
        """Get the set of host edges used by embedding mapping of tree tree_idx."""
        edges = set()
        for u, v in trees[tree_idx]:
            hu, hv = mapping[u], mapping[v]
            edges.add((min(hu, hv), max(hu, hv)))
        return edges

    def try_packing(order, rng_master):
        """
        Try to pack trees in the given order.
        Returns (mappings, value) or (None, 0) on timeout.
        """
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
                value += 0
                continue

            # First try greedy (fast)
            found = False
            rng = Random(rng_master * 10000 + tree_idx * 100)
            m = embed_tree_greedy(tree_idx, available, rng)
            if m is not None:
                mappings[tree_idx] = m
                used_edges |= get_used_edges(m, tree_idx)
                available = all_host_edges - used_edges
                value += len(tree_edges[tree_idx])
                found = True

            # If greedy failed, try backtracking
            if not found:
                for attempt in range(5):
                    if time_up():
                        break
                    rng = Random(rng_master * 10000 + tree_idx * 100 + 100 + attempt)
                    m = embed_tree_backtrack(tree_idx, available, rng)
                    if m is not None:
                        mappings[tree_idx] = m
                        used_edges |= get_used_edges(m, tree_idx)
                        available = all_host_edges - used_edges
                        value += len(tree_edges[tree_idx])
                        found = True
                        break

            if not found:
                mappings[tree_idx] = None

        return mappings, value

    # Strategy 1: Pack in descending order of tree size (T20 first, then T19, ...)
    order1 = list(range(19, -1, -1))

    # Strategy 2: Pack in ascending order
    order2 = list(range(20))

    # Strategy 3: Pack by difficulty (high max-degree first)
    max_deg = [max(tree_degrees[i]) if tree_degrees[i] else 0 for i in range(20)]
    order3 = sorted(range(20), key=lambda i: -max_deg[i])

    # Strategy 4: Alternate large and small to balance
    order4 = []
    large = [i for i in range(19, 9, -1)]  # T20..T10
    small = [i for i in range(9, -1, -1)]   # T10..T1
    for i in range(10):
        order4.append(large[i])
        order4.append(small[i])

    best_mappings = [None] * 20
    best_value = 0

    # Try each strategy with a few rng_master values
    strategies = [
        (order1, 1),
        (order1, 42),
        (order1, 123),
        (order2, 1),
        (order2, 42),
        (order3, 1),
        (order3, 42),
        (order3, 123),
        (order4, 1),
        (order4, 42),
    ]

    for order, rng_master in strategies:
        if time_up():
            break
        mappings, value = try_packing(order, rng_master)
        if value > best_value:
            best_value = value
            best_mappings = mappings

    # If we haven't achieved full packing, try a more intensive search on the best partial
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
                    best_value += 0
                    continue
                found = False
                # Try greedy first
                for attempt in range(20):
                    if time_up():
                        break
                    rng = Random(99999 + tree_idx * 1000 + attempt)
                    m = embed_tree_greedy(tree_idx, available, rng)
                    if m is not None:
                        best_mappings[tree_idx] = m
                        used_edges |= get_used_edges(m, tree_idx)
                        available = all_host_edges - used_edges
                        best_value += len(tree_edges[tree_idx])
                        found = True
                        break
                
                # Then try backtracking
                if not found:
                    for attempt in range(10):
                        if time_up():
                            break
                        rng = Random(88888 + tree_idx * 1000 + attempt)
                        m = embed_tree_backtrack(tree_idx, available, rng)
                        if m is not None:
                            best_mappings[tree_idx] = m
                            used_edges |= get_used_edges(m, tree_idx)
                            available = all_host_edges - used_edges
                            best_value += len(tree_edges[tree_idx])
                            found = True
                            break

    # Ensure all entries are either None or valid lists of ints
    result = []
    for i in range(20):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])

    return result
