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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
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


def _embed_with_backtracking(tree_edges, n, used_edges, rng, max_nodes=500000):
    """
    Embed tree_edges into K_n using backtracking with LCV heuristic and forward checking.
    Uses incremental free-neighbor tracking for speed.
    """
    if not tree_edges:
        return {}

    adj, degree, verts = _build_tree_info(tree_edges)
    k = len(verts)

    # Build edge_used as 2D boolean array
    edge_used = [[False] * n for _ in range(n)]
    for u, v in used_edges:
        edge_used[u][v] = True
        edge_used[v][u] = True

    # Precompute initial free degree for each host vertex
    free_deg = [0] * n
    for h in range(n):
        for x in range(n):
            if h != x and not edge_used[h][x]:
                free_deg[h] += 1

    # Sort vertices by degree descending for root candidates
    root_candidates = sorted(verts, key=lambda v: -degree[v])

    max_attempts = 100
    for attempt in range(max_attempts):
        # Pick a root with some randomization
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

        # Children map and subtree sizes
        children = {v: [] for v in order}
        for v in order[1:]:
            p = parent[v]
            children[p].append(v)
        
        # Compute subtree sizes
        subtree_size = {}
        for v in reversed(order):
            sz = 1
            for c in children[v]:
                sz += subtree_size[c]
            subtree_size[v] = sz

        mapping = {}
        host_used = [False] * n
        # Incremental free neighbor count: free_nbr[h] = number of x not host_used, not edge_used[h][x]
        free_nbr = free_deg[:]  # copy
        node_count = [0]

        def backtrack(pos):
            if node_count[0] > max_nodes:
                return False
            node_count[0] += 1

            if pos == len(order):
                return True

            v = order[pos]
            p = parent[v]
            child_list = children[v]
            num_children = len(child_list)

            if p == -1:
                candidates = [h for h in range(n) if not host_used[h]]
            else:
                hp = mapping[p]
                candidates = [h for h in range(n)
                              if not host_used[h]
                              and not edge_used[hp][h]]

            if not candidates:
                return False

            # Score: prefer host vertices with more free neighbors
            # and enough free neighbors for all children
            scored = []
            for h in candidates:
                fn = free_nbr[h]
                # Each child needs at least 1 distinct free neighbor
                if num_children > 0 and fn < num_children:
                    score = -1000000 + fn
                else:
                    score = fn
                scored.append((score, h))
            
            scored.sort(key=lambda x: -x[0])
            candidates = [h for _, h in scored]

            # Add randomization: shuffle top candidates
            if len(candidates) > 1:
                top_k = min(5, len(candidates))
                top = candidates[:top_k]
                rng.shuffle(top)
                candidates = top + candidates[top_k:]

            for h in candidates:
                mapping[v] = h
                host_used[h] = True
                
                # Update free_nbr for all other vertices: they lose h as a free neighbor
                saved = []
                for x in range(n):
                    if x != h and not host_used[x] and not edge_used[h][x]:
                        saved.append(x)
                        free_nbr[x] -= 1
                
                # Forward check: verify each child has at least one available neighbor
                feasible = True
                if num_children > 0:
                    if free_nbr[h] < num_children:
                        feasible = False
                    else:
                        # Hall's condition check for children (simplified)
                        # Each child needs a distinct free neighbor of h
                        # This is always satisfiable if free_nbr[h] >= num_children
                        # since we just need distinct vertices
                        pass
                
                if feasible and backtrack(pos + 1):
                    return True
                
                # Undo free_nbr updates
                for x in saved:
                    free_nbr[x] += 1
                
                del mapping[v]
                host_used[h] = False

            return False

        if backtrack(0):
            return dict(mapping)

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    trees = instance_trees(20, 611)
    n = 20

    # Process trees in order of decreasing constraint (most constrained first)
    def tree_constraint(i):
        if not trees[i]:
            return (0, 0, 0)
        adj_t = defaultdict(list)
        for u, v in trees[i]:
            adj_t[u].append(v)
            adj_t[v].append(u)
        max_deg = max((len(adj_t[v]) for v in adj_t), default=0)
        return (len(trees[i]), max_deg, i)
    
    order = list(range(20))
    order.sort(key=lambda i: tree_constraint(i), reverse=True)

    # Try multiple global random seeds
    best_total = 0
    best_result = [None] * 20

    for global_seed in range(2000):
        if time.time() > deadline - 5.0:
            break

        rng = Random(global_seed)
        used_edges = set()
        result = [None] * 20
        success_count = 0

        for idx in order:
            if time.time() > deadline - 3.0:
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
