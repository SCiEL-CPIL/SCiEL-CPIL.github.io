"""Rewritten program: attempt exact decomposition of K20 into trees T1..T20 using backtracking."""
from random import Random
import sys
sys.setrecursionlimit(100000)

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: 
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
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
    
    # Precompute tree adjacency and degree
    tree_adj = []
    for i in range(n):
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in trees[i]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Order trees: Largest first (most edges)
    order = list(range(n))
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    best_value = 0
    best_result = [None] * n
    
    # Try multiple random restarts
    for trial in range(100):
        rng = Random(trial)
        used_edges = set()
        result = [None] * n
        success = True
        
        for idx in order:
            k = idx + 1
            if k == 1:
                # Any vertex
                result[idx] = [rng.randint(0, n-1)]
                continue
                
            adj = tree_adj[idx]
            # We will use a recursive backtracking to find an embedding
            # State: mapping from tree_node to host_node
            mapping = [-1] * k
            host_used = [False] * n # For this tree only
            
            # Heuristic: Process tree nodes in an order. 
            # Let's just use 0..k-1 but with random shuffling of candidates.
            # To be more effective, we can pick the next node to assign as the one 
            # with the most already-assigned neighbors.
            
            def dfs(node_idx, depth):
                if depth == k:
                    return True
                
                # Select next node to assign
                # Find unassigned node with most assigned neighbors
                best_node = -1
                max_assigned_neighbors = -1
                for v in range(k):
                    if mapping[v] != -1:
                        continue
                    cnt = 0
                    for nb in adj[v]:
                        if mapping[nb] != -1:
                            cnt += 1
                    if cnt > max_assigned_neighbors:
                        max_assigned_neighbors = cnt
                        best_node = v
                
                if best_node == -1:
                    # Should not happen if depth < k, but just in case
                    best_node = next(v for v in range(k) if mapping[v] == -1)

                candidates = []
                for h in range(n):
                    if host_used[h]:
                        continue
                    # Check edges to already assigned neighbors
                    valid = True
                    for nb in adj[best_node]:
                        if mapping[nb] != -1:
                            h_nb = mapping[nb]
                            edge = (min(h, h_nb), max(h, h_nb))
                            if edge in used_edges:
                                valid = False
                                break
                    if valid:
                        candidates.append(h)
                
                rng.shuffle(candidates)
                
                for h in candidates:
                    mapping[best_node] = h
                    host_used[h] = True
                    if dfs(0, depth + 1): # node_idx not used in this logic, just depth
                        return True
                    mapping[best_node] = -1
                    host_used[h] = False
                
                return False

            if dfs(0, 0):
                # Success
                result[idx] = mapping[:]
                # Update used_edges
                for u, v in trees[idx]:
                    hu = mapping[u]
                    hv = mapping[v]
                    edge = (min(hu, hv), max(hu, hv))
                    used_edges.add(edge)
            else:
                success = False
                break
        
        if success:
            value = sum(len(trees[i]) for i in range(n) if result[i] is not None)
            if value > best_value:
                best_value = value
                best_result = result[:]
                if value == 190:
                    break
    
    if best_result is not None and best_value > 19:
        out = [None] * n
        for i in range(n):
            if best_result[i] is not None:
                # Ensure types are Python int
                out[i] = [int(v) for v in best_result[i]]
        return out
    
    # Fallback: Return a valid partial solution or empty
    # The task says "Entry k-1 is None or a list...". 
    # If we fail, we should return something valid, even if low score.
    # The seed program returned [None]*19 + [list(range(n))] which is likely invalid 
    # because T20 has 19 edges and K20 has 190, but T20 is a tree on 20 vertices.
    # A single tree T20 embedded in K20 uses 19 edges. This is a valid partial solution.
    # Let's try to embed just the largest tree as a fallback.
    
    # Actually, if best_result is all None, we can try to embed T20 alone.
    if best_value == 0:
        # Try to embed T20 (index 19)
        idx = 19
        k = 20
        adj = tree_adj[idx]
        mapping = [-1] * k
        host_used = [False] * n
        rng = Random(42)
        
        def dfs_fallback(node_idx, depth):
            if depth == k:
                return True
            best_node = -1
            max_assigned_neighbors = -1
            for v in range(k):
                if mapping[v] != -1:
                    continue
                cnt = 0
                for nb in adj[v]:
                    if mapping[nb] != -1:
                        cnt += 1
                if cnt > max_assigned_neighbors:
                    max_assigned_neighbors = cnt
                    best_node = v
            if best_node == -1:
                best_node = next(v for v in range(k) if mapping[v] == -1)
            
            candidates = [h for h in range(n) if not host_used[h]]
            rng.shuffle(candidates)
            
            for h in candidates:
                valid = True
                for nb in adj[best_node]:
                    if mapping[nb] != -1:
                        h_nb = mapping[nb]
                        # No used_edges check for fallback single tree, just injectivity
                        # But we need to ensure it's a valid embedding of the tree structure
                        # which is guaranteed by the tree structure.
                        pass
                # For a single tree, any injective mapping is a valid embedding 
                # as long as we don't care about edge disjointness with others (there are none).
                # Wait, the tree edges must map to distinct host edges. 
                # Since the tree is a tree, and mapping is injective, 
                # edges (u,v) in tree map to (H_u, H_v) in host. 
                # Are (H_u, H_v) distinct for different (u,v)? 
                # Yes, because if (H_u, H_v) == (H_x, H_y) with {u,v} != {x,y}, 
                # then {H_u, H_v} = {H_x, H_y}. Since mapping is injective, {u,v}={x,y}.
                # So any injective mapping works for a single tree.
                mapping[best_node] = h
                host_used[h] = True
                if dfs_fallback(0, depth + 1):
                    return True
                mapping[best_node] = -1
                host_used[h] = False
            return False
        
        if dfs_fallback(0, 0):
            best_result[idx] = mapping[:]
            best_value = len(trees[idx])
            out = [None] * n
            out[idx] = [int(v) for v in best_result[idx]]
            return out
            
    return [None] * n
