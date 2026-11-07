"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    n = 20
    trees = instance_trees(n, 611)
    
    # All edges of K_n
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    used_edges = set()
    result = []
    
    for k in range(1, n + 1):
        tree_edges = trees[k - 1]
        num_tree_edges = len(tree_edges)
        
        if num_tree_edges == 0:
            result.append(None)
            continue
        
        # We need to find an injection from tree vertices {0..k-1} to host vertices {0..n-1}
        # such that for every (u, v) in tree_edges, the mapped edge (f(u), f(v)) is in all_edges \ used_edges
        
        # This is a constraint satisfaction problem. We can use backtracking with pruning.
        # Since k <= 20 and n = 20, we're essentially looking for a permutation (or partial permutation if k < n)
        # that satisfies the edge constraints.
        
        # For k = n = 20, we need a bijection (permutation) of 20 vertices.
        # For k < n, we need an injection.
        
        # Let's use a simple backtracking approach with some heuristics.
        # Order the tree vertices by their degree in the tree (higher degree first) to constrain early.
        
        # Build adjacency for the tree
        tree_adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Sort vertices by degree (descending) for better pruning
        vertices_by_degree = sorted(range(k), key=lambda v: -len(tree_adj[v]))
        
        # We'll assign host vertices to tree vertices in this order.
        # assigned[tree_v] = host_v, or -1 if unassigned
        assigned = [-1] * k
        host_used = [False] * n
        
        # For efficiency, we can precompute for each tree vertex which host vertices are "compatible"
        # but compatibility depends on already-assigned neighbors.
        
        def backtrack(idx):
            if idx == k:
                return True
            
            tree_v = vertices_by_degree[idx]
            
            # Try each available host vertex
            for host_v in range(n):
                if host_used[host_v]:
                    continue
                
                # Check if assigning host_v to tree_v is consistent with already-assigned neighbors
                valid = True
                for neighbor in tree_adj[tree_v]:
                    if assigned[neighbor] != -1:
                        # Check if the edge between tree_v and neighbor maps to a valid, unused edge
                        e = (min(assigned[neighbor], host_v), max(assigned[neighbor], host_v))
                        if e not in all_edges or e in used_edges:
                            valid = False
                            break
                        # Also check that this edge hasn't been "claimed" by another tree edge already
                        # We'll track claimed edges during backtracking
                        if e in claimed_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Assign
                assigned[tree_v] = host_v
                host_used[host_v] = True
                
                # Mark edges to assigned neighbors as claimed
                newly_claimed = []
                for neighbor in tree_adj[tree_v]:
                    if assigned[neighbor] != -1:
                        e = (min(assigned[neighbor], host_v), max(assigned[neighbor], host_v))
                        claimed_edges.add(e)
                        newly_claimed.append(e)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                for e in newly_claimed:
                    claimed_edges.remove(e)
                host_used[host_v] = False
                assigned[tree_v] = -1
            
            return False
        
        claimed_edges = set()
        
        if backtrack(0):
            # Build the result: for each tree vertex i, what's its host vertex
            mapping = assigned[:]
            result.append(mapping)
            
            # Mark used edges
            for u, v in tree_edges:
                e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                used_edges.add(e)
        else:
            # Fallback: try a simpler approach or return None
            result.append(None)
    
    return result
# EVOLVE-BLOCK-END
