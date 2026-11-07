"""Rewritten program: Deterministic Backtracking with Forward Checking for K20 decomposition."""
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
    
    # Precompute tree adjacency
    tree_adj = []
    for i in range(n):
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in trees[i]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Order trees: Largest first
    order = list(range(n))
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # Global state
    used_edges = set()
    result = [None] * n
    
    # We will use a recursive function for each tree embedding
    # State: 
    #   mapping: list of host vertices for tree nodes (or -1)
    #   domains: list of sets of available host vertices for each tree node
    
    def embed_tree(tree_idx, adj, k):
        # Initialize domains
        # All nodes can initially map to any host vertex
        # But we need to handle injectivity and edge disjointness dynamically
        # For simplicity in this implementation, we will use a standard backtracking 
        # with MRV and Forward Checking on the domains.
        
        # To make it efficient, we precompute the list of neighbors for each tree node.
        
        # We process nodes in a dynamic order determined by MRV (Minimum Remaining Values).
        # We maintain a stack of assignments.
        
        # For performance, we will use a recursive helper that maintains:
        # - mapping: list[int]
        # - domains: list[set]
        
        # However, copying domains is expensive. 
        # Given n=20, k<=20, the search space is small enough for simple copying 
        # if we are careful, or we can use trail-based undo.
        # Let's use simple copying for clarity, as k is small.
        
        mapping = [-1] * k
        domains = [set(range(n)) for _ in range(k)]
        
        def backtrack(depth):
            if depth == k:
                return True
            
            # MRV: Select unassigned node with smallest domain size
            min_size = n + 1
            best_node = -1
            for v in range(k):
                if mapping[v] == -1:
                    if len(domains[v]) < min_size:
                        min_size = len(domains[v])
                        best_node = v
                        if min_size == 0:
                            return False # Pruned
            
            if best_node == -1:
                return True # Should not happen if depth < k and logic is correct
            
            # Value Ordering: Try values in domains[best_node]
            # Heuristic: Try values that leave largest domains for neighbors (Least Constraining Value)
            # For now, sorted order for determinism, or we can shuffle for randomization if stuck.
            # Let's use sorted order for deterministic behavior.
            values = sorted(domains[best_node])
            
            for h in values:
                # Check if h is valid for best_node
                # 1. h is in domain (implied)
                # 2. Check edge conflicts with already assigned neighbors
                valid = True
                for nb in adj[best_node]:
                    if mapping[nb] != -1:
                        h_nb = mapping[nb]
                        edge = (min(h, h_nb), max(h, h_nb))
                        if edge in used_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Assign
                mapping[best_node] = h
                domains[best_node].discard(h) # Remove from domain? No, we track assigned via mapping.
                
                # Forward Checking: Update domains of unassigned neighbors
                undo_list = []
                failed = False
                for nb in adj[best_node]:
                    if mapping[nb] == -1:
                        # Remove h from domains[nb] (injectivity)
                        if h in domains[nb]:
                            domains[nb].discard(h)
                            undo_list.append((nb, h))
                        
                        # Remove any host h' such that (h, h') is a used edge
                        # And h' is in domains[nb]
                        to_remove = []
                        for h_prime in domains[nb]:
                            edge = (min(h, h_prime), max(h, h_prime))
                            if edge in used_edges:
                                to_remove.append(h_prime)
                        
                        for h_prime in to_remove:
                            domains[nb].discard(h_prime)
                            undo_list.append((nb, h_prime))
                        
                        if len(domains[nb]) == 0:
                            failed = True
                            break
                
                if not failed:
                    if backtrack(depth + 1):
                        return True
                
                # Undo
                for nb, val in undo_list:
                    domains[nb].add(val)
                mapping[best_node] = -1
        
        if backtrack(0):
            return mapping[:]
        return None

    # Iterate through trees in order
    success = True
    for idx in order:
        k = idx + 1
        if k == 1:
            # Any vertex, pick 0 for determinism
            result[idx] = [0]
            continue
            
        adj = tree_adj[idx]
        mapping = embed_tree(idx, adj, k)
        
        if mapping is None:
            success = False
            break
        
        result[idx] = mapping
        # Update used_edges
        for u, v in trees[idx]:
            hu = mapping[u]
            hv = mapping[v]
            edge = (min(hu, hv), max(hu, hv))
            used_edges.add(edge)
            
    if success:
        # Convert to list of ints
        out = [None] * n
        for i in range(n):
            if result[i] is not None:
                out[i] = [int(v) for v in result[i]]
        return out
    
    # Fallback if deterministic fails (should be rare if solution exists)
    # Try random restarts as a backup, similar to original but with better logic
    for trial in range(50):
        rng = Random(trial)
        used_edges_set = set()
        res = [None] * n
        ok = True
        for idx in order:
            k = idx + 1
            if k == 1:
                res[idx] = [rng.randint(0, n-1)]
                continue
            adj = tree_adj[idx]
            mapping = [-1] * k
            host_used = [False] * n
            
            def dfs_rand(node_idx, depth):
                if depth == k:
                    return True
                best_node = -1
                max_assigned = -1
                for v in range(k):
                    if mapping[v] != -1: continue
                    cnt = sum(1 for nb in adj[v] if mapping[nb] != -1)
                    if cnt > max_assigned:
                        max_assigned = cnt
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
                            e = (min(h, h_nb), max(h, h_nb))
                            if e in used_edges_set:
                                valid = False
                                break
                    if valid:
                        mapping[best_node] = h
                        host_used[h] = True
                        if dfs_rand(0, depth + 1):
                            return True
                        mapping[best_node] = -1
                        host_used[h] = False
                return False
            
            if dfs_rand(0, 0):
                res[idx] = mapping[:]
                for u, v in trees[idx]:
                    e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                    used_edges_set.add(e)
            else:
                ok = False
                break
        
        if ok:
            out = [None] * n
            for i in range(n):
                if res[i] is not None:
                    out[i] = [int(v) for v in res[i]]
            return out
            
    return [None] * n
