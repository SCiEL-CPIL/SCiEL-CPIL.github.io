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
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 20
    trees = instance_trees(N, 611)
    
    # Precompute edge indices for bitmask operations
    # edge_idx[u][v] is the bit position for edge (u, v)
    edge_idx = [[-1] * N for _ in range(N)]
    edge_count = 0
    for u in range(N):
        for v in range(u + 1, N):
            edge_idx[u][v] = edge_count
            edge_idx[v][u] = edge_count
            edge_count += 1
            
    # Sort trees by size descending to pack large ones first
    # tree_order[i] is the index in the original 'trees' list
    tree_order = sorted(range(N), key=lambda i: len(trees[i]), reverse=True)
    
    used_mask = 0
    placements = [None] * N
    
    # Helper to check if edge (u, v) is available
    def is_edge_available(u, v):
        idx = edge_idx[u][v]
        return (used_mask >> idx) & 1 == 0
        
    # Helper to mark edge (u, v) as used
    def mark_edge_used(u, v):
        nonlocal used_mask
        idx = edge_idx[u][v]
        used_mask |= (1 << idx)
        
    # Helper to mark edge (u, v) as unused
    def mark_edge_unused(u, v):
        nonlocal used_mask
        idx = edge_idx[u][v]
        used_mask &= ~(1 << idx)

    def embed_tree(tree_idx, rng):
        """
        Try to embed tree 'tree_idx' into the remaining available edges.
        Returns a list of host vertices for the tree vertices, or None.
        """
        edges = trees[tree_idx]
        k = len(edges) + 1 if edges else 1
        if k == 1:
            # Any vertex
            return [0]
            
        # Build adjacency list for the tree
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Root the tree at 0
        # Create a DFS order to assign vertices
        visited = [False] * k
        order = []
        parent = [-1] * k
        
        def dfs(u):
            visited[u] = True
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    parent[v] = u
                    dfs(v)
                    
        dfs(0)
        
        mapping = [None] * k
        host_used = [False] * N
        
        # Heuristic: Try to assign host vertices that have high available degree?
        # Or just iterate.
        # To break symmetry for the first tree, we can fix mapping[0] = 0.
        # But for generality, we let the backtracking handle it.
        # However, fixing the first vertex reduces search space by N.
        # Let's fix mapping[order[0]] = 0 if it's the first tree being placed?
        # Actually, we can fix the first assigned vertex to 0 for ANY tree if we are allowed to permute the host graph?
        # No, the host graph is fixed. But the problem is symmetric under permutation of host vertices.
        # So for the FIRST tree placed, we can fix its root to 0.
        # For subsequent trees, we cannot fix as aggressively because the available edges are asymmetric.
        
        # Let's use a simple backtracking.
        # We assign vertices in 'order'.
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                return True
                
            tv = order[pos]
            
            # Get candidate host vertices
            # If tv has a parent already assigned, the host vertex must be connected to the parent's host via an available edge.
            if parent[tv] != -1:
                p_tv = parent[tv]
                h_p = mapping[p_tv]
                # Candidates must be neighbors of h_p with available edge
                candidates = [h for h in range(N) if not host_used[h] and is_edge_available(h, h_p)]
            else:
                # Root, any unused host vertex
                candidates = [h for h in range(N) if not host_used[h]]
                
            # Heuristic: Sort candidates by available degree (descending) to fill "holes" or (ascending)?
            # Usually, "fail first" suggests picking the most constrained.
            # Let's just try them. Randomizing the order helps with restarts.
            rng.shuffle(candidates)
            
            for h in candidates:
                # Check if h is compatible with other already assigned neighbors?
                # In a tree, a vertex only has one parent. Other neighbors are children (unassigned).
                # So we only need to check the edge to the parent.
                # Wait, what if the tree has cycles? No, it's a tree.
                # So the only constraint from already assigned vertices is the parent edge.
                
                mapping[tv] = h
                host_used[h] = True
                # Mark the edge to parent as used temporarily?
                # We don't need to mark it in 'used_mask' yet, because 'used_mask' is global for all trees.
                # But we need to ensure we don't use the same edge twice WITHIN this tree.
                # Since it's a tree, and we assign parent->child, the edge (parent, child) is unique.
                # And we check `is_edge_available` against the global `used_mask` which contains edges from PREVIOUS trees.
                # We do NOT add the current tree's edges to `used_mask` until the whole tree is placed.
                # So we must ensure intra-tree edge distinctness.
                # Since we check `is_edge_available` which only checks global used_mask, we might pick an edge
                # that is used by another edge in the SAME tree?
                # No, because the tree is a tree, and we are assigning a mapping.
                # If two different tree edges map to the same host edge, that would imply a cycle in the host graph
                # corresponding to a path in the tree?
                # Actually, if (u,v) and (u,w) in tree map to (a,b) and (a,b), that means v=w, which is not possible as v,w are distinct tree vertices.
                # So intra-tree conflicts are impossible for simple trees if the mapping is a bijection.
                
                if backtrack(pos + 1):
                    return True
                    
                mapping[tv] = None
                host_used[h] = False
                
            return False
            
        # Optimization: For the very first tree, fix the root to 0 to break symmetry
        # This reduces the search space significantly.
        # We identify if this is the first tree by checking if used_mask == 0.
        if used_mask == 0:
            # Fix order[0] to 0
            mapping[order[0]] = 0
            host_used[0] = True
            if backtrack(1):
                return mapping
            # If fixing to 0 fails, try other?
            # Due to symmetry, if it fails with 0, it fails with any.
            # So return None.
            return None
        else:
            if backtrack(0):
                return mapping
            return None

    # Main loop
    # We try to embed trees in descending order of size.
    # If a tree fails, we try a few random restarts for that tree before giving up.
    
    success_count = 0
    for t_idx in tree_order:
        if time.time() > deadline:
            break
            
        placed = False
        # Try up to 50 random restarts for this tree
        for attempt in range(50):
            if time.time() > deadline:
                break
            rng = Random(attempt * 1000 + t_idx)
            mapping = embed_tree(t_idx, rng)
            if mapping is not None:
                # Verify and commit
                valid = True
                for u, v in trees[t_idx]:
                    if not is_edge_available(mapping[u], mapping[v]):
                        valid = False
                        break
                if valid:
                    for u, v in trees[t_idx]:
                        mark_edge_used(mapping[u], mapping[v])
                    placements[t_idx] = [int(x) for x in mapping]
                    placed = True
                    success_count += 1
                    break
                else:
                    # Should not happen if logic is correct
                    pass
        
        # If not placed, we leave it as None.
        # We continue to try smaller trees.

    return placements
