import time
import random
import sys

sys.setrecursionlimit(100000)

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    deadline = start_time + 145

    trees = instance_trees(20, 611)
    n_trees = 20
    n_vertices = 20

    # Precompute tree adjacencies and degrees
    tree_adj = []
    tree_edges = []
    tree_deg = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj_list = [[] for _ in range(k)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        tree_adj.append(adj_list)
        tree_edges.append(edges)
        tree_deg.append([len(adj_list[v]) for v in range(k)])

    # Sort trees by size descending, then by max degree descending
    tree_indices = list(range(n_trees))
    tree_indices.sort(key=lambda x: (-(x + 1), -max(tree_deg[x]) if tree_deg[x] else 0))

    best_score = 0
    best_mappings = [None] * n_trees

    def try_packing(attempt_seed):
        nonlocal best_score, best_mappings
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * n_trees

        def pack_tree(tree_order_idx):
            t_idx = tree_indices[tree_order_idx]
            k = t_idx + 1
            adj_t = tree_adj[t_idx]

            if k == 1:
                mappings[t_idx] = [0]
                return True

            # Order vertices: high-degree first, then BFS from highest degree
            root = max(range(k), key=lambda v: (tree_deg[t_idx][v], -v))

            # Build ordering: root first, then neighbors in degree order
            order = [root]
            visited = {root}
            frontier = [root]
            while frontier:
                next_frontier = []
                for u in frontier:
                    neighbors = sorted(adj_t[u], key=lambda v: (-tree_deg[t_idx][v], v))
                    for nb in neighbors:
                        if nb not in visited:
                            visited.add(nb)
                            order.append(nb)
                            next_frontier.append(nb)
                frontier = next_frontier

            # Precompute which tree vertices are adjacent to which in the ordering
            # For forward checking: for each position, which later positions are neighbors
            pos_of_tv = [0] * k
            for i, tv in enumerate(order):
                pos_of_tv[tv] = i

            # Backtracking search with forward checking
            def backtrack(pos, mapping, used_verts):
                if time.time() > deadline:
                    return False

                if pos == k:
                    # All vertices mapped - commit
                    for u, v in tree_edges[t_idx]:
                        hu = mapping[u]
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            return False
                    for u, v in tree_edges[t_idx]:
                        hu = mapping[u]
                        hv = mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                    mappings[t_idx] = mapping[:]
                    return True

                tv = order[pos]
                mapped_tv = set(order[:pos])

                # Get candidates: unused host vertices that don't conflict with mapped neighbors
                candidates = []
                for hv in range(n_vertices):
                    if hv in used_verts:
                        continue
                    # Check edges to already-mapped neighbors
                    conflict = False
                    for nb in adj_t[tv]:
                        if nb in mapped_tv:
                            hn = mapping[nb]
                            e = (min(hv, hn), max(hv, hn))
                            if e in used_edges:
                                conflict = True
                                break
                    if not conflict:
                        candidates.append(hv)

                if not candidates:
                    return False

                # Forward checking: for each candidate, verify that all unmapped
                # neighbors still have at least one available host vertex
                # Also count available neighbors for ordering
                scored_candidates = []
                for hv in candidates:
                    fc_ok = True
                    total_avail = 0
                    for nb in adj_t[tv]:
                        if nb not in mapped_tv:
                            # For this neighbor, count how many host vertices are still available
                            avail = 0
                            for hnb in range(n_vertices):
                                if hnb in used_verts or hnb == hv:
                                    continue
                                # Check if edge (hv, hnb) is used
                                e1 = (min(hv, hnb), max(hv, hnb))
                                if e1 in used_edges:
                                    continue
                                # Also check hnb doesn't conflict with other mapped neighbors of nb
                                h_conflict = False
                                for nb2 in adj_t[nb]:
                                    if nb2 in mapped_tv and nb2 != tv:
                                        hn2 = mapping[nb2]
                                        e2 = (min(hnb, hn2), max(hnb, hn2))
                                        if e2 in used_edges:
                                            h_conflict = True
                                            break
                                if not h_conflict:
                                    avail += 1
                            if avail == 0:
                                fc_ok = False
                                break
                            total_avail += avail
                    if fc_ok:
                        scored_candidates.append((total_avail, hv))

                if not scored_candidates:
                    return False

                # Sort by fewest available neighbors first (most constrained)
                scored_candidates.sort()

                # Add small randomness for diversity
                if local_rng.random() < 0.2:
                    local_rng.shuffle(scored_candidates)

                for _, hv in scored_candidates:
                    mapping[tv] = hv
                    used_verts.add(hv)
                    if backtrack(pos + 1, mapping, used_verts):
                        return True
                    used_verts.remove(hv)
                    mapping[tv] = -1

                return False

            # Try different starting vertices for the root
            host_candidates = list(range(n_vertices))
            local_rng.shuffle(host_candidates)

            for h0 in host_candidates:
                mapping = [-1] * k
                mapping[order[0]] = h0
                used_verts = {h0}

                if backtrack(1, mapping, used_verts):
                    return True

            return False

        # Main loop
        success = True
        for i in range(n_trees):
            if time.time() > deadline:
                success = False
                break
            if pack_tree(i):
                pass
            else:
                success = False
                break

        # Calculate score
        score = 0
        for i in range(n_trees):
            if mappings[i] is not None:
                score += len(tree_edges[i])

        if score > best_score:
            best_score = score
            best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]

        return success

    # Run multiple attempts with different seeds
    attempt = 0
    while time.time() < deadline:
        try_packing(attempt)
        attempt += 1
        if best_score == 190:
            break

    return best_mappings
