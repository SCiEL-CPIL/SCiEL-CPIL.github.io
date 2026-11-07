"""Packing trees T1..T20 into K20 via backtracking with edge-disjointness constraint."""

import time
from random import Random, sample
from itertools import product

# Deadline guard: leave 5s margin before the 150s kill
_DEADLINE = 145.0
_t0 = time.time()


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
    global _t0
    _t0 = time.time()
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Sort trees by number of edges (descending) for better pruning
    # T_k has k-1 edges, so T20 has 19 edges, T19 has 18, etc.
    # But T1 has 0 edges (trivially embeddable)
    # Sort by complexity: larger trees first
    order = list(range(n))  # indices 0..19 for T1..T20
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # Edge-disjointness: K20 has C(20,2) = 190 edges
    # Total tree edges = sum(k-1 for k=1..20) = sum(0..19) = 190
    # So we need a perfect packing: every edge of K20 used exactly once!
    
    # Strategy: backtracking. For each tree (in order), find an injective
    # vertex mapping such that all tree edges map to unused K20 edges.
    
    # Precompute: for each tree, its edges in terms of vertex labels
    tree_edges = [trees[i] for i in range(n)]
    
    # Used edges set
    used_edges = set()
    
    # Result: mapping for each tree index
    result = [None] * n
    
    # For each tree, we need to assign its k vertices to k distinct host vertices
    # such that for each edge (u,v) in the tree, the host edge (map[u], map[v]) is unused.
    
    # Backtracking approach:
    # For tree T_k with k vertices, try all injective mappings (k! possibilities after
    # fixing one vertex, but we can use symmetry).
    
    # Better: use the structure. For small k, enumerate. For large k, use heuristics.
    
    # Since total edges = 190 = edges in K20, this is an edge-decomposition problem.
    # Each tree T_k must use exactly k-1 edges, and all 190 edges are partitioned.
    
    # Approach: recursive backtracking with memoization on used-edge sets is too large.
    # Instead, try a greedy + local search approach, or direct backtracking with good ordering.
    
    # Let's try: embed trees in order from largest to smallest.
    # For each tree, find a valid embedding (injective vertex map) using backtracking.
    
    def time_left():
        return _DEADLINE - (time.time() - _t0)
    
    def try_embed_tree(tree_idx, used_edges_set):
        """Try to find an embedding of tree_idx given current used_edges_set.
        Returns a list of k host vertices (mapping) or None if failed."""
        k = tree_idx + 1
        if k <= 1:
            return [0] if k == 1 else None  # T1: single vertex, map to any; use 0
        if k == 2:
            # Two vertices, one edge. Find any unused edge (a,b).
            # Return [a, b]
            for a in range(n):
                for b in range(a + 1, n):
                    e = (min(a, b), max(a, b))
                    if e not in used_edges_set:
                        return [a, b]
            return None
        
        # For k >= 3: backtracking
        edges = tree_edges[tree_idx]
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find a root: use vertex 0 as root, BFS order
        # We'll assign host vertices in BFS order of the tree
        parent = [-1] * k
        order_v = [0]
        parent[0] = -2  # root
        for u in list(order_v):
            for v in adj[u]:
                if parent[v] == -1:
                    parent[v] = u
                    order_v.append(v)
        
        # Backtracking: assign host vertices in order_v
        def bt(pos, host_map, used_set):
            if time_left() < 0.5:
                return None
            if pos == k:
                # Check all edges are valid (they should be by construction if we check during assignment)
                # Actually we check as we go
                return dict(host_map)
            
            v = order_v[pos]
            # Try each available host vertex
            for h in range(n):
                if h in host_map.values():
                    continue
                # Check edges from v to already-assigned neighbors
                ok = True
                for nb in adj[v]:
                    if parent[nb] == v:
                        continue  # child, not yet assigned
                    if nb == parent[v] or (parent[v] == -2 and nb != v):
                        # nb is already assigned (it's a parent or earlier sibling? No, only parent)
                        # In BFS, only parent is assigned before v (children come after)
                        pass
                    # Check if nb is already assigned
                    if nb in host_map:
                        e = (min(h, host_map[nb]), max(h, host_map[nb]))
                        if e in used_set:
                            ok = False
                            break
                        # Also check this edge is not already used by THIS tree
                        # (can't happen since tree is a tree, no multiedges)
                if not ok:
                    continue
                # Tentatively assign
                host_map[v] = h
                new_used = used_set.copy()
                for nb in adj[v]:
                    if nb in host_map:
                        e = (min(h, host_map[nb]), max(h, host_map[nb]))
                        new_used.add(e)
                
                result = bt(pos + 1, host_map, new_used)
                if result is not None:
                    return result
                
                del host_map[v]
            
            return None
        
        return bt(0, {}, used_edges_set)
    
    # Main backtracking over trees
    def solve_trees(idx, used_edges_set, result_list):
        if time_left() < 1.0:
            return False
        
        if idx == len(order):
            return True
        
        t_idx = order[idx]
        k = t_idx + 1
        
        # Try to embed tree t_idx
        emb = try_embed_tree(t_idx, used_edges_set)
        if emb is None:
            return False
        
        # Verify embedding
        edges = tree_edges[t_idx]
        for (u, v) in edges:
            e = (min(emb[u], emb[v]), max(emb[u], emb[v]))
            if e in used_edges_set:
                return False  # should not happen
            used_edges_set.add(e)
        
        result_list[t_idx] = [emb[i] for i in range(k)]
        
        if solve_trees(idx + 1, used_edges_set, result_list):
            return True
        
        # Undo
        for (u, v) in edges:
            e = (min(emb[u], emb[v]), max(emb[u], emb[v]))
            used_edges_set.discard(e)
        result_list[t_idx] = None
        
        return False
    
    # Try solving
    used_set = set()
    res = [None] * n
    if solve_trees(0, used_set, res):
        # Convert to proper format
        final = []
        for i in range(n):
            if res[i] is None:
                final.append(None)
            else:
                # Ensure all are Python ints
                final.append([int(x) for x in res[i]])
        return final
    
    # Fallback: only embed T20
    return [None] * 19 + [list(range(20))]
