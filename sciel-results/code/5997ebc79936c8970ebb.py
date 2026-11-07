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
    deadline = time.time() + 140  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    n = 20
    
    # Pack trees in order of decreasing size (largest first) to constrain early
    # But we need to return results indexed by tree position (k-1 for tree k)
    # Let's pack in reverse order: T20 first, then T19, ..., T1
    # Actually, let's try packing largest first for better constraint propagation
    
    # Order: pack T20, T19, T18, ..., T1 (indices 19, 18, ..., 0)
    pack_order = list(range(19, -1, -1))
    
    # State: for each tree index, store the embedding (list of host vertices) or None
    result = [None] * 20
    # Used edges set
    used_edges = set()
    
    # Precompute adjacency for K20
    # We'll track available degree per vertex and used edges
    
    def get_available_neighbors(v, used_edges):
        """Get neighbors of v not yet used in an edge with v."""
        return [u for u in range(n) if u != v and (min(v,u), max(v,u)) not in used_edges]
    
    def backtrack(idx_in_order, used_edges, result):
        if time.time() > deadline:
            return False
        if idx_in_order == len(pack_order):
            return True
        
        tree_idx = pack_order[idx_in_order]
        tree = trees[tree_idx]
        k = tree_idx + 1  # tree T_k has k vertices
        
        if k == 1:
            result[tree_idx] = [0]  # single vertex, any vertex works
            return backtrack(idx_in_order + 1, used_edges, result)
        
        if k == 2:
            # Need one edge
            for u in range(n):
                for v in range(u+1, n):
                    edge = (u, v)
                    if edge not in used_edges:
                        result[tree_idx] = [u, v]
                        used_edges.add(edge)
                        if backtrack(idx_in_order + 1, used_edges, result):
                            return True
                        used_edges.remove(edge)
                        result[tree_idx] = None
            return False
        
        # For larger trees, use backtracking on vertex assignments
        # The tree has k vertices (0..k-1) and k-1 edges
        # We need to assign each tree vertex to a host vertex 0..19
        # such that all edges map to unused edges in K20
        
        # Build adjacency of the tree
        tree_adj = [[] for _ in range(k)]
        for (a, b) in tree:
            tree_adj[a].append(b)
            tree_adj[b].append(a)
        
        # Find a good root: highest degree vertex
        deg = [len(tree_adj[i]) for i in range(k)]
        root = max(range(k), key=lambda i: deg[i])
        
        # Do DFS from root to get an ordering of vertices
        visited = [False] * k
        order = [root]
        visited[root] = True
        stack = [root]
        while stack:
            u = stack.pop()
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order.append(v)
                    stack.append(v)
        
        # Backtrack over vertex assignments in the DFS order
        host = [None] * k  # host[tree_vertex] = host_vertex
        
        def assign_vertex(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                # All vertices assigned, check all edges are valid (should be by construction)
                return True
            
            tv = order[pos]
            for hv in range(n):
                # Check if hv can be assigned to tree vertex tv
                valid = True
                for neighbor_tv in tree_adj[tv]:
                    if host[neighbor_tv] is not None:
                        nhv = host[neighbor_tv]
                        edge = (min(hv, nhv), max(hv, nhv))
                        if edge in used_edges:
                            valid = False
                            break
                if not valid:
                    continue
                
                host[tv] = hv
                if assign_vertex(pos + 1):
                    return True
                host[tv] = None
            
            return False
        
        if assign_vertex(0):
            result[tree_idx] = [int(host[i]) for i in range(k)]
            # Add all tree edges to used_edges
            for (a, b) in tree:
                edge = (min(host[a], host[b]), max(host[a], host[b]))
                used_edges.add(edge)
            if backtrack(idx_in_order + 1, used_edges, result):
                return True
            # Undo
            for (a, b) in tree:
                edge = (min(host[a], host[b]), max(host[a], host[b]))
                used_edges.discard(edge)
            result[tree_idx] = None
            return False
        
        return False
    
    if backtrack(0, used_edges, result):
        # Convert to proper types
        final_result = []
        for i in range(20):
            if result[i] is None:
                final_result.append(None)
            else:
                final_result.append([int(x) for x in result[i]])
        return final_result
    else:
        # Fallback: return what we have (shouldn't happen if backtracking succeeds)
        final_result = []
        for i in range(20):
            if result[i] is None:
                final_result.append(None)
            else:
                final_result.append([int(x) for x in result[i]])
        return final_result
