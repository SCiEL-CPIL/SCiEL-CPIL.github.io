import time
import sys
from random import Random
from collections import Counter

def _generate_trees(n=20, seed=611):
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
    deadline = time.time() + 145
    trees = _generate_trees(20, 611)
    n = 20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Precompute tree structures
    tree_adj = []
    tree_leaves = []
    for tree_idx in range(20):
        k = tree_idx + 1
        if k == 1:
            tree_adj.append([])
            tree_leaves.append([0])
            continue
        adj = [[] for _ in range(k)]
        for (u, v) in trees[tree_idx]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        leaves = [v for v in range(k) if len(adj[v]) == 1]
        if not leaves:
            leaves = [0]
        tree_leaves.append(leaves)

    order = list(range(19, -1, -1))
    result = [None] * 20

    def can_embed(tree_idx, available_edges, rng):
        k = tree_idx + 1
        if k == 1:
            for v in range(n):
                return [v]
            return None
        if k == 2:
            for (a, b) in available_edges:
                return [a, b]
            return None

        tree_edges = trees[tree_idx]
        adj = tree_adj[tree_idx]
        leaves = tree_leaves[tree_idx]

        # Build available adjacency
        avail_adj = [set() for _ in range(n)]
        for (a, b) in available_edges:
            avail_adj[a].add(b)
            avail_adj[b].add(a)

        # Precompute degrees for heuristic ordering
        avail_deg = [len(avail_adj[v]) for v in range(n)]

        def try_root(leaf):
            parent = [-1] * k
            children = [[] for _ in range(k)]
            visited = [False] * k
            visited[leaf] = True
            stack = [leaf]
            order_list = [leaf]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        children[u].append(v)
                        stack.append(v)
                        order_list.append(v)

            host = [-1] * k
            used = set()
            # Track edge usage to avoid rechecking at the end
            used_edges = set()

            def backtrack(idx):
                if time.time() > deadline:
                    return None
                if idx == len(order_list):
                    # All vertices assigned; verify edges are available
                    # (should be guaranteed by incremental checks, but double-check)
                    return list(host)

                v = order_list[idx]
                if parent[v] == -1:
                    # Root: try all vertices, ordered by degree in available graph
                    candidates = list(range(n))
                    # Sort by degree (low first for MRV-like effect on root? or high?)
                    # Usually low degree is more constrained, but for root it's about flexibility.
                    # Let's stick to low degree first as it was before, but maybe high degree is better for root?
                    # Let's try sorting by degree descending (high degree first) for root to leave low degree for leaves?
                    # Actually, MRV says pick the most constrained variable first. Here we are assigning values to a fixed variable.
                    # The heuristic is on the value (host vertex).
                    # Let's try sorting by degree ascending (low degree vertices are more "special" or constrained).
                    candidates.sort(key=lambda h: avail_deg[h])
                    # Add some randomization to avoid getting stuck
                    rng.shuffle(candidates)
                else:
                    pv = parent[v]
                    if host[pv] == -1:
                        return None
                    # Candidates must be neighbors of parent in available graph
                    # And the edge (parent_host, h) must be available
                    candidates = []
                    ph = host[pv]
                    for h in avail_adj[ph]:
                        # Edge (ph, h) must be in available_edges
                        e = (min(ph, h), max(ph, h))
                        if e in available_edges and e not in used_edges:
                            candidates.append(h)
                    
                    # Sort candidates by degree (MRV: low degree first)
                    candidates.sort(key=lambda h: avail_deg[h])
                    rng.shuffle(candidates)

                for h in candidates:
                    if h in used:
                        continue
                    
                    # Forward checking: ensure all children of v still have options
                    # A child c of v needs a neighbor of h that is not used
                    # and the edge (h, neighbor) is available
                    feasible = True
                    for child in children[v]:
                        # Check if there is at least one available neighbor for this child
                        found_option = False
                        for nh in avail_adj[h]:
                            if nh in used:
                                continue
                            e = (min(h, nh), max(h, nh))
                            if e in available_edges and e not in used_edges:
                                found_option = True
                                break
                        if not found_option:
                            feasible = False
                            break
                    if not feasible:
                        continue

                    # Also, if this is not the root, the edge from parent to h is now used
                    edge_to_add = None
                    if parent[v] != -1:
                        ph = host[parent[v]]
                        edge_to_add = (min(ph, h), max(ph, h))
                        if edge_to_add in used_edges:
                            continue # Should have been filtered in candidates, but safety

                    host[v] = h
                    used.add(h)
                    if edge_to_add:
                        used_edges.add(edge_to_add)
                    
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    used.discard(h)
                    if edge_to_add:
                        used_edges.discard(edge_to_add)
                    host[v] = -1
                return None

            return backtrack(0)

        # Try different roots
        shuffled_leaves = leaves[:]
        rng.shuffle(shuffled_leaves)
        for leaf in shuffled_leaves:
            emb = try_root(leaf)
            if emb is not None:
                return emb
        return None

    best_result = [None] * 20
    best_score = -1
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        rng = Random(611 + attempt * 1000)
        result = [None] * 20
        available = set(all_edges)
        success = True

        for t_idx in order:
            if time.time() > deadline:
                success = False
                break
            k = t_idx + 1
            if k == 1:
                # Pick a random vertex? Or just 0?
                # Since T1 is a single vertex, it uses no edges.
                # We can assign it to any vertex.
                result[t_idx] = [0]
                continue
            elif k == 2:
                found = False
                # Pick a random edge from available
                if available:
                    # Convert to list for random choice? Or just pick first?
                    # Picking first is deterministic. Let's pick random.
                    edge_list = list(available)
                    if edge_list:
                        (a, b) = rng.choice(edge_list)
                        result[t_idx] = [a, b]
                        available.discard((a, b))
                        found = True
                if not found:
                    success = False
                    break
                continue

            emb = can_embed(t_idx, available, rng)
            if emb is None:
                success = False
                break
            result[t_idx] = emb
            for (u, v) in trees[t_idx]:
                hu, hv = emb[u], emb[v]
                e = (min(hu, hv), max(hu, hv))
                if e in available:
                    available.discard(e)
                else:
                    success = False
                    break
            if not success:
                break

        if success:
            # Calculate score: number of embedded edges
            # Since we succeeded, it's 190
            score = 190
            if score > best_score:
                best_score = score
                best_result = [list(r) if r is not None else None for r in result]
                # If we got a perfect score, we can break early?
                # The max possible is 190.
                if best_score == 190:
                    break
        else:
            # Maybe we can keep the partial result if it's better than current best?
            # But partial results are invalid if not all trees are embedded?
            # The task says: "Entry k-1 is None or a list... Embedded trees must be edge-disjoint"
            # It doesn't explicitly say ALL must be embedded, but "Packing trees T1..T20" implies all.
            # However, "raw value is the total number of embedded tree edges".
            # If some are None, they contribute 0 edges.
            # So a partial solution is valid if the embedded ones are edge-disjoint.
            # Let's calculate the score for partial success.
            score = 0
            valid = True
            used_edges = set()
            for i in range(20):
                if result[i] is None:
                    continue
                k = i + 1
                if k == 1:
                    continue
                if k == 2:
                    a, b = result[i]
                    e = (min(a, b), max(a, b))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                    score += 1
                else:
                    for (u, v) in trees[i]:
                        hu, hv = result[i][u], result[i][v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            valid = False
                            break
                        used_edges.add(e)
                        score += 1
                    if not valid:
                        break
            
            if valid and score > best_score:
                best_score = score
                best_result = [list(r) if r is not None else None for r in result]

    # Ensure types are correct
    for i in range(20):
        if best_result[i] is not None:
            best_result[i] = [int(x) for x in best_result[i]]

    return best_result
