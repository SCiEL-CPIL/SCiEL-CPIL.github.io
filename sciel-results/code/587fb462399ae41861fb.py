"""Packing trees T1..T20 into K20."""
import time
from random import Random
from itertools import permutations

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

def _embed_tree(tree_edges, used_edges, n=20):
    """
    Try to embed a tree with given edges into K_n avoiding used_edges.
    Returns a list of length len(tree_vertices) mapping tree-vertex -> host-vertex,
    or None if impossible.
    """
    if not tree_edges:
        return []
    
    # Determine the set of vertices in the tree
    verts = set()
    for u, v in tree_edges:
        verts.add(u)
        verts.add(v)
    verts = sorted(verts)
    k = len(verts)
    
    # Build adjacency for the tree
    adj = {v: [] for v in verts}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Choose a root - pick a vertex with max degree for better pruning
    root = max(verts, key=lambda v: len(adj[v]))
    
    # DFS order of the tree
    order = []
    parent = {}
    stack = [root]
    parent[root] = -1
    while stack:
        node = stack.pop()
        order.append(node)
        for nb in adj[node]:
            if nb != parent[node]:
                parent[nb] = node
                stack.append(nb)
    
    # Map tree vertex index to position in order
    # We'll assign host vertices in DFS order
    # For each tree vertex in order, try all available host vertices
    
    host_assignment = {}
    
    def backtrack(idx):
        if idx == len(order):
            return True
        tv = order[idx]
        # Determine constraints: if tv has a parent in the tree, the host must be adjacent
        # to the host of the parent (which is already assigned)
        # Also, the edge (host_parent, host_tv) must not be in used_edges
        
        if parent[tv] != -1:
            ph = host_assignment[parent[tv]]
            candidates = []
            for h in range(n):
                if h in host_assignment.values():
                    continue
                # Check edge not used
                edge = frozenset((ph, h))
                if edge in used_edges:
                    continue
                candidates.append(h)
        else:
            # Root: can be any unused host vertex
            candidates = [h for h in range(n) if h not in host_assignment.values()]
        
        for h in candidates:
            host_assignment[tv] = h
            if backtrack(idx + 1):
                return True
            del host_assignment[tv]
        return False
    
    if not backtrack(0):
        return None
    
    # Build result: map each tree vertex to its host
    result = [0] * k
    for tv in verts:
        idx_in_verts = verts.index(tv)
        result[idx_in_verts] = host_assignment[tv]
    
    # But we need the result to be indexed by tree vertex value (0..k-1)
    # Actually, the tree vertices are 0..k-1 based on how they're generated
    # Let me verify: in instance_trees, vertices are 0..k-1
    # So verts should be [0, 1, ..., k-1]
    
    # Return list where position i is the image of vertex i
    return [host_assignment[v] for v in range(k)]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin
    
    trees = instance_trees(20, 611)
    
    # Sort trees by size descending for better packing (harder first)
    # Actually, let's try embedding in order of decreasing complexity
    # But the contract says entry k-1 corresponds to T_k
    # We need to embed all trees, but we can choose which to embed first
    
    # Strategy: embed trees in order of decreasing number of edges (larger first)
    # This is because larger trees are harder to fit
    
    order = list(range(20))
    # Sort by number of edges descending
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    used_edges = set()
    results = [None] * 20
    
    for idx in order:
        if time.time() > deadline:
            break
        tree = trees[idx]
        if not tree:
            results[idx] = []
            continue
        embedding = _embed_tree(tree, used_edges, 20)
        if embedding is None:
            # Try a few random restarts with different root choices
            found = False
            for attempt in range(50):
                if time.time() > deadline:
                    break
                # Shuffle the order slightly
                tree_copy = list(tree)
                embedding = _embed_tree(tree_copy, used_edges, 20)
                if embedding is not None:
                    found = True
                    break
            if not found:
                continue
        results[idx] = embedding
        # Mark edges as used
        for u, v in tree:
            edge = frozenset((embedding[u], embedding[v]))
            used_edges.add(edge)
    
    # Convert to proper format
    final = []
    for i in range(20):
        if results[i] is None:
            final.append(None)
        else:
            # Ensure it's a list of ints
            final.append([int(x) for x in results[i]])
    
    return final
