"""Packing trees T1..T20 into K20 with edge-disjoint embeddings."""
import time
import random
from random import Random

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
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    trees = instance_trees(20, 611)
    n_trees = len(trees)
    n_hosts = 20
    
    # All possible edges in K20
    all_edges = set()
    for i in range(n_hosts):
        for j in range(i + 1, n_hosts):
            all_edges.add((i, j))
    
    # We'll try to pack all trees
    # Strategy: process trees in a good order, use backtracking
    
    def try_pack(tree_order, rng):
        """Try to pack trees in given order. Returns list of embeddings or None."""
        used_edges = set()
        embeddings = [None] * n_trees
        
        for idx in tree_order:
            k = idx + 1
            tree_edges = trees[idx]
            
            if k == 1:
                # Single vertex, any host vertex works
                embeddings[idx] = [0]
                continue
            
            if k == 2:
                # Single edge, find any unused edge
                found = False
                for _ in range(1000):
                    u = rng.randint(0, n_hosts - 1)
                    v = rng.randint(0, n_hosts - 1)
                    if u != v:
                        e = (min(u, v), max(u, v))
                        if e not in used_edges:
                            embeddings[idx] = [u, v]
                            used_edges.add(e)
                            found = True
                            break
                if not found:
                    # Try all pairs
                    for i in range(n_hosts):
                        for j in range(i + 1, n_hosts):
                            if (i, j) not in used_edges:
                                embeddings[idx] = [i, j]
                                used_edges.add((i, j))
                                found = True
                                break
                        if found:
                            break
                if not found:
                    return None
                continue
            
            # For k >= 3, use backtracking
            embedding = _find_embedding(k, tree_edges, used_edges, n_hosts, rng, deadline)
            if embedding is None:
                return None
            
            embeddings[idx] = embedding
            for u, v in tree_edges:
                e = (min(embedding[u], embedding[v]), max(embedding[u], embedding[v]))
                used_edges.add(e)
        
        return embeddings
    
    def _find_embedding(k, tree_edges, used_edges, n_hosts, rng, deadline):
        """Find an embedding for a tree with k vertices."""
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Root the tree at vertex 0
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        
        for node in order:
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = node
                    order.append(nb)
        
        # Backtracking: assign host vertices in BFS order
        host_assign = [-1] * k
        host_assign[0] = rng.randint(0, n_hosts - 1)
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            
            if pos == k:
                return True
            
            node = order[pos]
            # Try each available host vertex
            candidates = list(range(n_hosts))
            rng.shuffle(candidates)
            
            for hv in candidates:
                if any(host_assign[i] == hv for i in range(pos) if host_assign[i] >= 0):
                    continue
                
                # Check if edges to already-assigned neighbors are valid
                valid = True
                for nb in adj[node]:
                    if host_assign[nb] >= 0:
                        e = (min(hv, host_assign[nb]), max(hv, host_assign[nb]))
                        if e in used_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                host_assign[node] = hv
                if backtrack(pos + 1):
                    return True
                host_assign[node] = -1
            
            return False
        
        # Multiple attempts with different random starts
        for attempt in range(50):
            if time.time() > deadline:
                return None
            host_assign = [-1] * k
            host_assign[0] = rng.randint(0, n_hosts - 1)
            if backtrack(1):
                return host_assign[:]
        
        return None
    
    # Try different orderings and random seeds
    best_result = None
    best_value = -1
    
    # First, try a natural order with multiple random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        rng = Random(restart * 12345 + 678)
        
        # Try packing in natural order
        tree_order = list(range(n_trees))
        result = try_pack(tree_order, rng)
        
        if result is not None:
            # Count total edges
            total = 0
            for idx in range(n_trees):
                if result[idx] is not None:
                    total += len(trees[idx])
            
            if total > best_value:
                best_value = total
                best_result = result[:]
        
        # If we got all 190, we're done
        if best_value >= 190:
            break
    
    # If we didn't get a full packing, try to maximize what we can
    if best_result is None:
        # Fall back to just T20
        best_result = [None] * 19 + [list(range(20))]
        best_value = 19
    
    # Convert to proper format
    final = []
    for idx in range(n_trees):
        if best_result[idx] is None:
            final.append(None)
        else:
            final.append([int(v) for v in best_result[idx]])
    
    return final
