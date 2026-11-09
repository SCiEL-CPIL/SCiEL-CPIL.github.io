"""Packing trees T1..T20 into K20 using backtracking edge-disjoint embedding."""
import time
from random import Random
from collections import defaultdict


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


def _build_tree_info(tree_edges):
    """Build adjacency and degree info for a tree."""
    if not tree_edges:
        return {}, {}, []
    verts = set()
    for u, v in tree_edges:
        verts.add(u)
        verts.add(v)
    verts = sorted(verts)
    adj = {v: [] for v in verts}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    degree = {v: len(adj[v]) for v in verts}
    return adj, degree, verts


def _embed_with_backtracking(tree_edges, n, used_edges, rng, max_nodes=200000):
    """
    Embed tree_edges into K_n using backtracking with LCV heuristic and forward checking.
    Uses boolean arrays for fast edge/vertex availability checks.
    """
    if not tree_edges:
        return {}

    adj, degree, verts = _build_tree_info(tree_edges)
    k = len(verts)

    # Try different roots: prefer high-degree vertices first, then leaves
    high_deg = [v for v in verts if degree[v] >= 3]
    leaves = [v for v in verts if degree[v] == 1]
    mid = [v for v in verts if degree[v] == 2]
    
    root_candidates = []
    if high_deg:
        root_candidates.extend(high_deg)
    if mid:
        root_candidates.extend(mid)
    if leaves:
        root_candidates.extend(leaves)
    if not root_candidates:
        root_candidates = verts[:]

    # Precompute edge availability as a 2D boolean matrix
    edge_avail = [[False] * n for _ in range(n)]
    for h in range(n):
        for x in range(n):
            if h != x and (min(h, x), max(h, x)) not in used_edges:
                edge_avail[h][x] = True

    for attempt in range(30):
        if attempt < len(root_candidates):
            root = root_candidates[attempt % len(root_candidates)]
        else:
            root = rng.choice(root_candidates)

        # Build BFS order from root
        order = [root]
        parent = {root: -1}
        visited = {root}
        queue = [root]
        idx = 0
        while idx < len(queue):
            u = queue[idx]
            idx += 1
            for w in adj[u]:
                if w not in visited:
                    visited.add(w)
                    parent[w] = u
                    queue.append(w)
                    order.append(w)

        # Build parent_index: for each position in order, what's the parent's position
        pos_of = {v: i for i, v in enumerate(order)}
        parent_pos = [-1] * len(order)
        for i in range(1, len(order)):
            parent_pos[i] = pos_of[parent[order[i]]]

        # Children counts per position
        child_counts = [0] * len(order)
        for i in range(1, len(order)):
            child_counts[parent_pos[i]] += 1

        mapping = [None] * len(order)
        host_used = [False] * n
        node_count = [0]

        def backtrack(pos):
            if node_count[0] > max_nodes:
                return False
            node_count[0] += 1

            if pos == len(order):
                return True

            v = order[pos]
            ppos = parent_pos[pos]
            n_children = child_counts[pos]

            # Build candidate list
            if ppos == -1:
                candidates = [h for h in range(n) if not host_used[h]]
            else:
                hp = mapping[ppos]
                candidates = [h for h in range(n)
                              if not host_used[h] and edge_avail[hp][h]]

            if not candidates:
                return False

            # LCV Heuristic: score by free neighbor count
            def heuristic_score(h):
                free_count = 0
                for x in range(n):
                    if not host_used[x] and edge_avail[h][x]:
                        free_count += 1
                if n_children > 0 and free_count < n_children:
                    return -1000000 + free_count
                return free_count

            candidates.sort(key=heuristic_score, reverse=True)

            # Add some randomization: shuffle top candidates slightly
            if len(candidates) > 1:
                top_k = min(5, len(candidates))
                top = candidates[:top_k]
                rng.shuffle(top)
                candidates = top + candidates[top_k:]

            for h in candidates:
                mapping[pos] = h
                host_used[h] = True
                
                # Forward check: verify children have available neighbors
                feasible = True
                if n_children > 0:
                    free_count = 0
                    for x in range(n):
                        if not host_used[x] and edge_avail[h][x]:
                            free_count += 1
                    if free_count < n_children:
                        feasible = False
                
                if feasible and backtrack(pos + 1):
                    return True
                
                host_used[h] = False
                mapping[pos] = None

            return False

        if backtrack(0):
            result = {}
            for i, v in enumerate(order):
                result[v] = mapping[i]
            return result

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    trees = instance_trees(20, 611)
    n = 20

    # Process trees in order of decreasing number of edges (most constrained first)
    # But we need to return in original order
    def tree_constraint(i):
        if not trees[i]:
            return (0, 0)
        adj_t = defaultdict(list)
        for u, v in trees[i]:
            adj_t[u].append(v)
            adj_t[v].append(u)
        max_deg = max((len(adj_t[v]) for v in adj_t), default=0)
        return (len(trees[i]), max_deg)
    
    order = list(range(20))
    order.sort(key=lambda i: tree_constraint(i), reverse=True)

    # Try multiple global random seeds
    best_total = 0
    best_result = [None] * 20

    for global_seed in range(300):
        if time.time() > deadline - 3.0:
            break

        rng = Random(global_seed)
        used_edges = set()
        result = [None] * 20
        success_count = 0

        for idx in order:
            if time.time() > deadline - 2.0:
                break

            k = idx + 1
            tree_edges = trees[idx]

            if not tree_edges:
                # Tree with 1 vertex (k=1)
                result[idx] = [0]
                success_count += 1
                continue

            mapping = _embed_with_backtracking(tree_edges, n, used_edges, rng)
            if mapping is not None:
                result_list = []
                for v in range(k):
                    result_list.append(int(mapping[v]))
                result[idx] = result_list
                success_count += 1

                for u, v in tree_edges:
                    hu, hv = mapping[u], mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    used_edges.add(e)

        # Calculate total edges embedded
        total_edges = sum(len(trees[i]) for i in range(20) if result[i] is not None)

        if total_edges > best_total:
            best_total = total_edges
            best_result = [list(r) if r is not None else None for r in result]

        # Early exit if we found a perfect packing (all 190 edges)
        if best_total >= 190:
            break

    return best_result
