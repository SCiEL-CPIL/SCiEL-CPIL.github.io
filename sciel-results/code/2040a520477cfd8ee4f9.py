"""Packing trees T1..T20 into K20 - backtracking with heuristics."""
from random import Random
import time

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
    deadline = time.time() + 140.0  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    n = 20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Sort trees: embed larger trees first (more constrained)
    # Tree k has k-1 edges, so T20 is most constrained
    order = list(range(20))  # index 0 is T1, index 19 is T20
    # Embed in order of decreasing size for better pruning
    order.sort(key=lambda idx: len(trees[idx]), reverse=True)
    
    used_edges = set()
    embeddings = [None] * 20
    
    # For each tree, we need to find an injective mapping of its vertices
    # to host vertices 0..19 such that all edges map to unused host edges.
    
    def find_embedding(tree_edges, k):
        """Find injective vertex mapping for tree with k vertices."""
        if k == 1:
            # Single vertex, any unused vertex works
            for v in range(n):
                if v not in used_vertices:
                    return [v]
            return None
        if k == 2:
            # One edge (0,1)
            for u in range(n):
                for w in range(n):
                    if u != w and (u, w) not in used_edges and u not in used_vertices and w not in used_vertices:
                        return [u, w]
            return None
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for a, b in tree_edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Root the tree at vertex 0
        # Do DFS to get a rooted tree structure
        parent = [-1] * k
        order_list = [0]
        visited = [False] * k
        visited[0] = True
        for v in order_list:
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = v
                    order_list.append(nb)
        
        # Map tree vertices in DFS order: assign host vertices
        # tree_vertex[i] = host vertex for tree vertex i
        assignment = [-1] * k
        
        def backtrack(pos, used_v):
            if time.time() > deadline:
                return False
            if pos == k:
                # Verify all edges map to unused host edges
                for a, b in tree_edges:
                    ea, eb = assignment[a], assignment[b]
                    edge = (min(ea, eb), max(ea, eb))
                    if edge in used_edges:
                        return False
                # Success
                return True
            
            tv = order_list[pos]
            # Try all candidate host vertices
            candidates = []
            for hv in range(n):
                if hv in used_v:
                    continue
                # Check edge to parent if parent is already assigned
                if parent[tv] != -1:
                    p_host = assignment[parent[tv]]
                    edge = (min(hv, p_host), max(hv, p_host))
                    if edge in used_edges:
                        continue
                candidates.append(hv)
            
            # Heuristic: try vertices that are more "central" or have more options
            for hv in candidates:
                assignment[tv] = hv
                used_v.add(hv)
                if backtrack(pos + 1, used_v):
                    return True
                used_v.remove(hv)
                assignment[tv] = -1
            
            return False
        
        used_v = set()
        assignment[0] = -1  # will be set in backtrack
        if backtrack(0, used_v):
            return assignment
        return None
    
    # We need to track used vertices globally
    used_vertices = set()
    
    def solve(idx):
        """Recursively solve: embed trees in order[idx:]."""
        if time.time() > deadline:
            return False
        if idx == len(order):
            return True
        
        tree_idx = order[idx]
        tree_edges = trees[tree_idx]
        k = tree_idx + 1
        
        if k == 1:
            # Find any free vertex
            for v in range(n):
                if v not in used_vertices:
                    embeddings[tree_idx] = [v]
                    used_vertices.add(v)
                    if solve(idx + 1):
                        return True
                    used_vertices.remove(v)
                    embeddings[tree_idx] = None
            return False
        
        # Try to find embedding
        result = find_embedding(tree_edges, k)
        if result is None:
            return False
        
        # Check if embedding is valid with current used_edges
        valid = True
        for a, b in tree_edges:
            ea, eb = result[a], result[b]
            edge = (min(ea, eb), max(ea, eb))
            if edge in used_edges:
                valid = False
                break
        if not valid:
            return False
        
        # Also check no vertex conflicts (shouldn't happen since find_embedding checks used_vertices)
        for v in result:
            if v in used_vertices:
                valid = False
                break
        if not valid:
            return False
        
        # Commit
        embeddings[tree_idx] = result
        for a, b in tree_edges:
            ea, eb = result[a], result[b]
            used_edges.add((min(ea, eb), max(ea, eb)))
        for v in result:
            used_vertices.add(v)
        
        if solve(idx + 1):
            return True
        
        # Undo
        for a, b in tree_edges:
            ea, eb = result[a], result[b]
            used_edges.remove((min(ea, eb), max(ea, eb)))
        for v in result:
            used_vertices.remove(v)
        embeddings[tree_idx] = None
        
        return False
    
    if solve(0):
        # Convert to proper format
        result = []
        for i in range(20):
            if embeddings[i] is None:
                result.append(None)
            else:
                result.append([int(v) for v in embeddings[i]])
        return result
    
    # Fallback: just embed T20
    return [None] * 19 + [list(range(20))]
