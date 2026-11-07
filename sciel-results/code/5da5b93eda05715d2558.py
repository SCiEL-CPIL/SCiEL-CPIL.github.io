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

    def embed_tree(tree_idx, available_edges, rng, timeout_sec=5.0):
        """
        Try to embed tree tree_idx into K20 using available_edges.
        Returns list of length k (tree_idx+1) mapping tree vertex i to host vertex,
        or None if not found.
        """
        k = tree_idx + 1
        if k == 1:
            # Single vertex: pick any host vertex
            return [0]
        if k == 2:
            # Need one edge
            for e in available_edges:
                return [e[0], e[1]]
            return None

        deg = tree_degrees[tree_idx]
        adj = tree_adj[tree_idx]

        # Order tree vertices by degree descending (high-degree first for pruning)
        order = sorted(range(k), key=lambda x: -deg[x])
        pos_in_order = [0] * k
        for i, v in enumerate(order):
            pos_in_order[v] = i

        assigned = [-1] * k  # assigned[i] = host vertex for order[i], -1 if unassigned
        used_hosts = set()

        # Track remaining degree capacity per host vertex
        # rem_deg[h] = number of unused edges incident to host h
        rem_deg = [0] * N
        for e in available_edges:
            rem_deg[e[0]] += 1
            rem_deg[e[1]] += 1

        # For pruning: for each tree vertex, we need deg[v] distinct neighbors
        # So host must have rem_deg >= deg[v] (roughly)

        # DFS with backtracking
        # Use iterative approach with explicit stack for deep recursion safety
        # But for k<=20, recursion depth is fine

        def dfs(idx):
            if time_up():
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

            # Forward checking: for each unassigned tree vertex w that is a neighbor of v,
            # after assigning h to v, w must still have at least deg[w] available edges
            # to unassigned hosts (including h if w-v is an edge)
            
            candidates = []
            for h in range(N):
                if h in used_hosts:
                    continue
                # Check edges to assigned neighbors
                valid = True
                for u, hu in assigned_neighbors:
                    edge = (min(h, hu), max(h, hu))
                    if edge not in available_edges:
                        valid = False
                        break
                if not valid:
                    continue
                # Degree constraint: host needs at least v_deg unused edges
                if rem_deg[h] < v_deg:
                    continue
                
                # Forward checking: verify unassigned neighbors can still be satisfied
                fc_ok = True
                for w in adj[v]:
                    if pos_in_order[w] >= idx:  # w is unassigned
                        # w needs deg[w] edges total; it will use 1 edge to h (since v-w is tree edge)
                        # plus edges to other already-assigned neighbors
                        # Count how many of w's neighbors are already assigned
                        assigned_w_neighbors = 0
                        for x in adj[w]:
                            if pos_in_order[x] < idx:
                                assigned_w_neighbors += 1
                        # w needs deg[w] - assigned_w_neighbors - 1 more edges to unassigned hosts
                        # (the -1 is for the edge to h/v)
                        # Actually w needs deg[w] total edges. One will go to h.
                        # The rest must go to other hosts.
                        # The available edges from h to other unassigned hosts matter.
                        # Simple check: rem_deg[h] - 1 >= (number of unassigned neighbors of w that need h)
                        # This is complex; use simpler check:
                        # After using edge (h, future_host_of_w), rem_deg[h] decreases by 1
                        # But we haven't assigned w yet, so just check basic feasibility
                        pass
                
                if fc_ok:
                    candidates.append(h)

            if not candidates:
                return False

            # Sort by rem_deg ascending (most constrained first), then break ties randomly
            candidates.sort(key=lambda h: rem_deg[h])
            rng.shuffle(candidates)

            for h in candidates:
                # Assign
                assigned[pos_in_order[v]] = h
                used_hosts.add(h)
                # Update rem_deg: h will use deg[v] edges total, but we track incrementally
                # Each edge from h to an assigned neighbor consumes 1 from rem_deg[h]
                # and 1 from rem_deg[neighbor]
                old_rem_h = rem_deg[h]
                rem_deg[h] -= len(assigned_neighbors)  # edges to already-assigned neighbors
                # Also decrement rem_deg for those neighbors
                old_rems = []
                for u, hu in assigned_neighbors:
                    old_rems.append(rem_deg[hu])
                    rem_deg[hu] -= 1

                # Try next
                if dfs(idx + 1):
                    return True

                # Undo
                assigned[pos_in_order[v]] = -1
                used_hosts.discard(h)
                rem_deg[h] = old_rem_h
                for i, (u, hu) in enumerate(assigned_neighbors):
                    rem_deg[hu] = old_rems[i]

            return False

        if dfs(0):
            # Construct mapping in original vertex order
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
                value += 0  # T1 has 0 edges
                continue

            # Try multiple rng seeds for this tree
            found = False
            for attempt in range(3):
                if time_up():
                    break
                rng = Random(rng_master * 10000 + tree_idx * 100 + attempt)
                m = embed_tree(tree_idx, available, rng)
                if m is not None:
                    mappings[tree_idx] = m
                    used_edges |= get_used_edges(m, tree_idx)
                    available = all_host_edges - used_edges
                    value += len(tree_edges[tree_idx])
                    found = True
                    break

            if not found:
                mappings[tree_idx] = None
                # Continue with smaller trees

        return mappings, value

    # Strategy 1: Pack in descending order of tree size (T20 first, then T19, ...)
    # This is good because larger trees are harder to fit
    order1 = list(range(19, -1, -1))

    # Strategy 2: Pack in ascending order
    order2 = list(range(20))

    # Strategy 3: Pack by difficulty (high max-degree first)
    max_deg = [max(tree_degrees[i]) if tree_degrees[i] else 0 for i in range(20)]
    order3 = sorted(range(20), key=lambda i: -max_deg[i])

    # Strategy 4: Pack by edge count (most edges first, which correlates with size)
    edge_counts = [len(tree_edges[i]) for i in range(20)]
    order4 = sorted(range(20), key=lambda i: -edge_counts[i])

    best_mappings = [None] * 20
    best_value = 0

    # Try each strategy with a few rng_master values
    strategies = [
        (order1, 1),
        (order1, 42),
        (order1, 123),
        (order4, 1),
        (order4, 42),
        (order4, 123),
        (order3, 1),
        (order3, 42),
        (order2, 1),
        (order2, 42),
    ]

    for order, rng_master in strategies:
        if time_up():
            break
        mappings, value = try_packing(order, rng_master)
        if value > best_value:
            best_value = value
            best_mappings = mappings
        # If we've achieved full packing, stop early
        if best_value >= 190:
            break

    # If we haven't achieved full packing, try a more intensive search on the best partial
    # Focus on filling in missing trees
    if best_value < 190 and not time_up():
        # Try to fill in missing trees with more attempts
        missing = [i for i in range(20) if best_mappings[i] is None]
        if missing:
            # Rebuild used_edges from current mappings
            used_edges = set()
            for i in range(20):
                if best_mappings[i] is not None:
                    used_edges |= get_used_edges(best_mappings[i], i)
            available = all_host_edges - used_edges

            # Try to embed missing trees, largest first
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
                for attempt in range(20):
                    if time_up():
                        break
                    rng = Random(99999 + tree_idx * 1000 + attempt)
                    m = embed_tree(tree_idx, available, rng)
                    if m is not None:
                        best_mappings[tree_idx] = m
                        used_edges |= get_used_edges(m, tree_idx)
                        available = all_host_edges - used_edges
                        best_value += len(tree_edges[tree_idx])
                        found = True
                        break
                
                # If still not found, try a different approach: temporarily remove
                # some already-placed trees and re-pack
                if not found and not time_up():
                    # Try removing the largest already-placed tree and re-packing
                    placed = [i for i in range(20) if best_mappings[i] is not None and i != tree_idx]
                    if placed:
                        # Remove the largest placed tree
                        remove_idx = max(placed, key=lambda i: (i + 1))
                        used_edges -= get_used_edges(best_mappings[remove_idx], remove_idx)
                        available = all_host_edges - used_edges
                        best_mappings[remove_idx] = None
                        best_value -= len(tree_edges[remove_idx])
                        
                        # Now try to embed both tree_idx and remove_idx
                        for t_idx in [tree_idx, remove_idx]:
                            if time_up():
                                break
                            tk = t_idx + 1
                            if tk == 1:
                                best_mappings[t_idx] = [0]
                                continue
                            for attempt in range(10):
                                if time_up():
                                    break
                                rng = Random(77777 + t_idx * 500 + attempt)
                                m = embed_tree(t_idx, available, rng)
                                if m is not None:
                                    best_mappings[t_idx] = m
                                    used_edges |= get_used_edges(m, t_idx)
                                    available = all_host_edges - used_edges
                                    best_value += len(tree_edges[t_idx])
                                    break

    # Ensure all entries are either None or valid lists of ints
    result = []
    for i in range(20):
        if best_mappings[i] is None:
            result.append(None)
        else:
            # Ensure it's a list of Python ints
            result.append([int(x) for x in best_mappings[i]])

    return result
