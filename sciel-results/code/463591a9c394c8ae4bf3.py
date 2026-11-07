"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    trees = instance_trees(20, 611)
    n = 20
    
    # Track used edges in K20
    used_edges = set()
    
    def get_used_edge(u, v):
        return (min(u, v), max(u, v))
    
    # Try to embed trees from largest to smallest (more constrained first)
    # But T1 and T2 are trivial, so handle them specially
    # Order: embed larger trees first to reduce options for smaller ones
    
    # Actually, let's try a different strategy:
    # For each tree, try all possible injective vertex mappings and pick one that doesn't conflict
    
    def try_embed_tree(tree_edges, n_hosts):
        """Try to find an injective mapping of tree vertices to host vertices such that no edge conflicts."""
        if not tree_edges:
            return list(range(len(tree_edges)))  # empty tree
        
        # Get unique vertices in the tree
        verts_in_tree = set()
        for u, v in tree_edges:
            verts_in_tree.add(u)
            verts_in_tree.add(v)
        verts_in_tree = sorted(verts_in_tree)
        k = len(verts_in_tree)
        
        if k == 0:
            return []
        if k == 1:
            # Single vertex, any host works
            for h in range(n_hosts):
                return [h]
        if k == 2:
            # Single edge
            for u in range(n_hosts):
                for v in range(n_hosts):
                    if u == v:
                        continue
                    e = get_used_edge(u, v)
                    if e not in used_edges:
                        return [u, v]
            return None
        
        # For larger trees, use backtracking with pruning
        # Map tree vertex indices to their position in verts_in_tree
        vertex_to_idx = {v: i for i, v in enumerate(verts_in_tree)}
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            iu = vertex_to_idx[u]
            iv = vertex_to_idx[v]
            adj[iu].append(iv)
            adj[iv].append(iu)
        
        # Try to find an injective mapping
        assignment = [None] * k
        
        def backtrack(idx):
            if idx == k:
                return True
            
            # Try each available host vertex
            for h in range(n_hosts):
                if h in assignment:
                    continue
                
                assignment[idx] = h
                valid = True
                
                # Check edges to already-assigned neighbors
                for neighbor in adj[idx]:
                    if assignment[neighbor] is not None:
                        e = get_used_edge(h, assignment[neighbor])
                        if e in used_edges:
                            valid = False
                            break
                
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                assignment[idx] = None
            
            return False
        
        if backtrack(0):
            return assignment
        return None
    
    # Process trees in order of decreasing size (more constrained first)
    # T1 (0 verts), T2 (2 verts), T3..T20
    order = list(range(20))
    # Sort by number of vertices descending, but keep T1, T2 at appropriate places
    # Actually, let's just go from largest to smallest tree
    order_sorted = sorted(order, key=lambda i: len(trees[i]) if i > 0 else 0, reverse=True)
    
    results = [None] * 20
    
    for idx in order_sorted:
        tree = trees[idx]
        if not tree:
            results[idx] = []
            continue
        
        # Try to embed
        mapping = try_embed_tree(tree, n)
        if mapping is None:
            # Fallback: try a simpler greedy
            # For now, mark as None
            results[idx] = None
            continue
        
        # Apply the mapping and mark edges as used
        vertex_to_idx = {v: i for i, v in enumerate(sorted(set(u for u, v in tree) | set(v for u, v in tree)))}
        
        for u, v in tree:
            iu = vertex_to_idx[u]
            iv = vertex_to_idx[v]
            hu = mapping[iu]
            hv = mapping[iv]
            e = get_used_edge(hu, hv)
            used_edges.add(e)
        
        results[idx] = mapping
    
    # Verify all are valid and count total edges
    total_edges = 0
    for idx in range(20):
        if results[idx] is None:
            continue
        tree = trees[idx]
        if not tree:
            continue
        vertex_to_idx = {v: i for i, v in enumerate(sorted(set(u for u, v in tree) | set(v for u, v in tree)))}
        for u, v in tree:
            iu = vertex_to_idx[u]
            iv = vertex_to_idx[v]
            hu = results[idx][iu]
            hv = results[idx][iv]
            if hu == hv:
                return [None] * 20  # invalid
            total_edges += 1
    
    return results
# EVOLVE-BLOCK-END
