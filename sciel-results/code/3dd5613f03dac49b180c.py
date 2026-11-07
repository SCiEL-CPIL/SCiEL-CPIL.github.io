"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 20
    trees = instance_trees(N, 611)
    
    # Build edge set for K20
    all_edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_edges.add((i, j))
    
    # Precompute tree data
    tree_data = []
    for idx in range(N):
        k = idx + 1
        edges = trees[idx]
        # Build adjacency
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Compute degrees
        deg = [0] * k
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        tree_data.append({'k': k, 'edges': edges, 'adj': adj, 'deg': deg})
    
    # Strategy: Process trees in order of decreasing size for better pruning
    # Use backtracking with the following approach:
    # For each tree, try all possible rootings and assignments
    
    # Key insight: We can use the fact that K20 is highly symmetric
    # to reduce the search space. Fix the embedding of the first few vertices.
    
    # Let's use a recursive backtracking solver
    # State: which edges are used, and which trees are placed
    
    used_edges = set()
    placement = [None] * N  # placement[i] = list of N host vertices for tree i, or None
    
    def time_check():
        return time.time() < deadline
    
    def try_place_tree(tree_idx):
        """
        Try to place tree_idx using unused edges.
        Returns list of host vertices (length k) or None.
        """
        if not time_check():
            return None
        
        td = tree_data[tree_idx]
        k = td['k']
        edges = td['edges']
        adj = td['adj']
        
        if k == 1:
            # Any single vertex works
            for v in range(N):
                return [v]
            return None
        
        if k == 2:
            for a in range(N):
                for b in range(a + 1, N):
                    if (a, b) not in used_edges:
                        return [a, b]
            return None
        
        # For larger trees, use backtracking
        # Root at vertex 0
        # Get BFS/DFS ordering
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
        
        # Precompute which tree vertices are adjacent
        # For pruning: when assigning a vertex, check all assigned neighbors
        
        def backtrack(pos):
            if not time_check():
                return False
            if pos == k:
                return True
            
            tv = order[pos]
            
            for hv in range(N):
                if host_used[hv]:
                    continue
                
                # Check all edges from tv to already-assigned vertices
                ok = True
                for nb in adj[tv]:
                    if mapping[nb] is not None:
                        a, b = min(hv, mapping[nb]), max(hv, mapping[nb])
                        if (a, b) in used_edges:
                            ok = False
                            break
                        # Also need to ensure this edge isn't going to be used twice
                        # within this tree - but since tree has no multi-edges, 
                        # and we're checking each edge once, this should be fine
                        # Actually, we need to check that no two tree edges map to the same host edge
                        # But since tree is simple and we're checking against used_edges (global),
                        # and we'll add all tree edges at once, we need to track intra-tree conflicts
                        # For now, assume the tree structure prevents this (it's a tree, no cycles)
                
                if not ok:
                    continue
                
                # Additional pruning: check if hv has enough unused edges to connect
                # to all unassigned neighbors of tv
                unassigned_neighbors = [nb for nb in adj[tv] if mapping[nb] is None]
                if unassigned_neighbors:
                    # hv must have at least len(unassigned_neighbors) unused edges
                    # to vertices that could be assigned to those neighbors
                    # This is complex, skip for now
                    pass

                mapping[tv] = hv
                host_used[hv] = True
                
                if backtrack(pos + 1):
                    return True
                
                mapping[tv] = None
                host_used[hv] = False
            
            return False
        
        if backtrack(0):
            # Verify the embedding
            for u, v in edges:
                a, b = min(mapping[u], mapping[v]), max(mapping[u], mapping[v])
                if (a, b) in used_edges:
                    return None  # Shouldn't happen
            return mapping
        
        return None
    
    def solve(pos):
        if not time_check():
            return False
        if pos == N:
            return True
        
        # Choose next tree to place - use dynamic variable ordering
        # Pick the tree with the most constraints (largest, or most connected to already placed)
        # For simplicity, use fixed order: largest first
        # But we can also try to pick the tree that has the most edges already constrained
        
        # Simple heuristic: process in order of decreasing tree size
        # Actually, let's just iterate through remaining trees
        for idx in range(N):
            if placement[idx] is not None:
                continue
            
            # Try to place this tree
            mapping = try_place_tree(idx)
            if mapping is not None:
                # Add edges to used
                td = tree_data[idx]
                for u, v in td['edges']:
                    a, b = min(mapping[u], mapping[v]), max(mapping[u], mapping[v])
                    used_edges.add((a, b))
                
                placement[idx] = mapping
                
                if solve(pos + 1):
                    return True
                
                # Undo
                for u, v in td['edges']:
                    a, b = min(mapping[u], mapping[v]), max(mapping[u], mapping[v])
                    used_edges.discard((a, b))
                
                placement[idx] = None
            
            break  # Only try one tree per level to keep it simple
        
        return False
    
    # Fixed order: largest trees first
    tree_order = sorted(range(N), key=lambda i: tree_data[i]['k'], reverse=True)
    
    def solve_ordered(pos):
        if not time_check():
            return False
        if pos == N:
            return True
        
        idx = tree_order[pos]
        mapping = try_place_tree(idx)
        if mapping is None:
            return False
        
        td = tree_data[idx]
        for u, v in td['edges']:
            a, b = min(mapping[u], mapping[v]), max(mapping[u], mapping[v])
            used_edges.add((a, b))
        
        placement[idx] = mapping
        
        if solve_ordered(pos + 1):
            return True
        
        for u, v in td['edges']:
            a, b = min(mapping[u], mapping[v]), max(mapping[u], mapping[v])
            used_edges.discard((a, b))
        
        placement[idx] = None
        return False
    
    success = solve_ordered(0)
    
    # If failed, try other orderings
    if not success:
        used_edges.clear()
        placement = [None] * N
        
        # Try ascending order
        tree_order = sorted(range(N), key=lambda i: tree_data[i]['k'])
        success = solve_ordered(0)
    
    if not success:
        used_edges.clear()
        placement = [None] * N
        
        # Try random orderings
        rng = Random(123)
        for attempt in range(100):
            if not time_check():
                break
            tree_order = list(range(N))
            rng.shuffle(tree_order)
            
            # Re-define solve_ordered for this order
            def solve_order2(pos):
                if not time_check():
                    return False
                if pos == N:
                    return True
                
                idx = tree_order[pos]
                mapping = try_place_tree(idx)
                if mapping is None:
                    return False
                
                td = tree_data[idx]
                for u, v in td['edges']:
                    a, b = min(mapping[u], mapping[v]), max(mapping[u], mapping[v])
                    used_edges.add((a, b))
                
                placement[idx] = mapping
                
                if solve_order2(pos + 1):
                    return True
                
                for u, v in td['edges']:
                    a, b = min(mapping[u], mapping[v]), max(mapping[u], mapping[v])
                    used_edges.discard((a, b))
                
                placement[idx] = None
                return False
            
            success = solve_order2(0)
            if success:
                break
    
    # Build result
    result = []
    for i in range(N):
        if placement[i] is None:
            result.append(None)
        else:
            result.append([int(v) for v in placement[i]])
    
    return result
# EVOLVE-BLOCK-END
