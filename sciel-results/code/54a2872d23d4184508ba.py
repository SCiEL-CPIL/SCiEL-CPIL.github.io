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
    deadline = time.time() + 140
    trees = instance_trees(20, 611)
    n = 20

    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Precompute tree info
    tree_info = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k <= 2:
            tree_info.append(None)
            continue
        tree_adj = {v: [] for v in range(k)}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        ordered_verts = sorted(range(k), key=lambda v: (-len(tree_adj[v]), v))
        tree_info.append((tree_edges, tree_adj, ordered_verts))

    def try_embed_rest(res, avail, start_idx, rng):
        """
        Try to embed trees from start_idx down to 0.
        res is a list of size 20, where res[t_idx] is the embedding for T_{t_idx+1}.
        avail is the set of available edges.
        """
        # We process from largest remaining to smallest
        # start_idx is the index of the next tree to embed (0-based index in trees list, so T_{start_idx+1})
        # We assume res[start_idx+1:] are already filled.
        
        # We need to embed trees[start_idx], trees[start_idx-1], ... trees[0]
        # Let's iterate downwards
        for t_idx in range(start_idx, -1, -1):
            k = t_idx + 1
            if k == 1:
                # T1 is a single vertex. Pick any available vertex.
                # But we must ensure it doesn't conflict? No, it uses no edges.
                # However, the vertex must be distinct from others? 
                # "Entry k-1 is None or a list of k distinct int host vertices"
                # "Embedded trees must be edge-disjoint"
                # Vertices can be shared between trees? 
                # "list of k distinct int host vertices" -> distinct *within* the tree.
                # Can T1's vertex be the same as T2's vertex? 
                # The problem doesn't forbid vertex sharing, only edge disjointness.
                # However, usually in such packings, vertex sharing is allowed.
                # Let's pick a random vertex.
                res[t_idx] = [rng.randint(0, n - 1)]
                continue
            
            if k == 2:
                # T2 is an edge. Pick any available edge.
                # To be deterministic and simple, pick the first one found.
                # But we should be careful.
                found = False
                for u, v in avail:
                    res[t_idx] = [u, v]
                    avail.discard((u, v))
                    found = True
                    break
                if not found:
                    return False
                continue

            # For k >= 3, use backtracking
            tree_edges, tree_adj, ordered_verts = tree_info[t_idx]
            host_map = {}
            used_hosts = set()

            # Optimization: Fix the first vertex of the tree to a specific host to break symmetry
            # This is valid because we can permute host labels.
            # However, the residual graph is not fully symmetric anymore.
            # But fixing one vertex reduces search space by factor 20.
            
            # Let's try a few fixed positions for the first vertex of the tree
            # We'll integrate this into the backtracking by iterating over a few candidate hosts for the first vertex.
            
            def backtrack(pos):
                if time.time() > deadline:
                    return False
                if pos == k:
                    return True

                tv = ordered_verts[pos]
                candidates = []

                # Determine candidate hosts
                # If pos == 0, we can restrict to a subset of hosts to break symmetry
                # e.g., hosts 0, 1, 2
                if pos == 0:
                    candidate_hosts = [0, 1, 2]
                else:
                    candidate_hosts = range(n)

                for h in candidate_hosts:
                    if h in used_hosts:
                        continue
                    ok = True
                    for nb in tree_adj[tv]:
                        if nb in host_map:
                            hn = host_map[nb]
                            edge = (min(h, hn), max(h, hn))
                            if edge not in avail:
                                ok = False
                                break
                    if ok:
                        # Heuristic: prefer hosts with fewer available edges
                        # This helps "fill in" the graph
                        remaining = 0
                        for j in range(n):
                            if j != h and (min(h, j), max(h, j)) in avail:
                                remaining += 1
                        candidates.append((remaining, h))

                if not candidates:
                    return False

                candidates.sort(key=lambda x: x[0])
                # Randomize top few
                if len(candidates) > 1:
                    top_n = min(len(candidates), 10)
                    top = candidates[:top_n]
                    rng.shuffle(top)
                    rest = candidates[top_n:]
                    candidates = top + rest

                for _, h in candidates:
                    host_map[tv] = h
                    used_hosts.add(h)
                    if backtrack(pos + 1):
                        return True
                    del host_map[tv]
                    used_hosts.remove(h)

                return False

            # Try to embed
            if backtrack(0):
                res[t_idx] = [host_map[i] for i in range(k)]
                # Update avail
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    avail.discard((min(hu, hv), max(hu, hv)))
            else:
                return False

        return True

    # Main loop
    # We will try to embed T20 first, then T19, etc.
    # We can use the symmetry breaking in try_embed_rest for all trees, or just T20.
    # Let's use it for all.
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
        rng = Random(attempt * 12345 + 67890)
        avail = set(all_edges)
        res = [None] * 20
        
        # Start with T20 (index 19)
        # We can just call try_embed_rest with start_idx=19
        # But try_embed_rest processes downwards from start_idx.
        # So we need to embed T20, then T19...
        # The function try_embed_rest assumes res[start_idx+1:] are filled.
        # So we call try_embed_rest(res, avail, 19, rng).
        
        if try_embed_rest(res, avail, 19, rng):
            # Check if all edges are used
            if len(avail) == 0:
                # Valid solution found
                return [None if r is None else [int(x) for x in r] for r in res]
            # If not all edges used, it's still a valid packing of the trees, 
            # but the score is based on "total number of embedded tree edges".
            # The trees are fixed, so if we embedded them, the edges are fixed.
            # The only way to have < 190 edges is if we failed to embed some trees?
            # No, if try_embed_rest returns True, all trees are embedded.
            # So the number of edges is always 190.
            # Why would avail not be empty?
            # Because the trees might not cover all edges of K20?
            # Sum of edges is 190. K20 has 190 edges.
            # So if we embed all trees disjointly, we MUST use all edges.
            # So len(avail) should be 0.
            # If it's not, something is wrong (e.g. duplicate edges in trees? No, they are trees).
            # So we can just return res.
            return [None if r is None else [int(x) for x in r] for r in res]

    # If no solution found
    return [None] * 20

