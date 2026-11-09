"""
Ramsey graph search: maximize n such that there exists an (n,5,5)-free graph.
Uses local search with simulated annealing on the number of K5 and independent-5
configurations, starting from the Paley graph on 37 vertices.
"""
import time
import random
import numpy as np

def _paley_graph(q=37):
    residues = {(x * x) % q for x in range(1, q)}
    adj = np.zeros((q, q), dtype=bool)
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i, j] = adj[j, i] = True
    return adj

def _count_k5_fast(adj, k=5):
    """
    Count number of K5 cliques in the whole graph using adjacency matrix.
    Optimized for n <= 42.
    """
    from itertools import combinations
    n = adj.shape[0]
    count = 0
    # Precompute adjacency as a list of lists of neighbors for faster access?
    # Or just use the matrix.
    # To speed up, we can iterate over pairs and count triangles in common neighbors.
    # Number of K5s = sum over edges (u,v) of (Number of triangles in N(u) intersect N(v)) / 10?
    # No, each K5 has 10 edges. Each K5 is counted 10 times (once for each edge).
    # So count = sum_{u<v, adj[u,v]} count_triangles(N(u) & N(v)) / 10.
    
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                common = np.logical_and(adj[u, :], adj[v, :])
                common[u] = False
                common[v] = False
                indices = np.where(common)[0]
                if len(indices) < 3:
                    continue
                sub_adj = adj[np.ix_(indices, indices)]
                # Count triangles in sub_adj
                # Vectorized triangle count for small subgraph
                tri_count = 0
                m = len(indices)
                for i in range(m):
                    for j in range(i + 1, m):
                        if sub_adj[i, j]:
                            # Common neighbors of i and j in sub_adj
                            # (i, j) in sub_adj correspond to indices[i], indices[j]
                            # We need to count k > j such that sub_adj[i, k] and sub_adj[j, k]
                            common_ij = np.logical_and(sub_adj[i, :], sub_adj[j, :])
                            common_ij[:j+1] = False
                            tri_count += np.sum(common_ij)
                total += tri_count
    return total // 10

def _count_independent_k5(adj, k=5):
    """Count independent sets of size k, i.e., K5 in the complement."""
    n = adj.shape[0]
    comp = ~adj
    np.fill_diagonal(comp, False)
    return _count_k5_fast(comp, k)

def _total_cost(adj):
    """Total number of K5 + independent 5-sets."""
    k5 = _count_k5_fast(adj, 5)
    ind5 = _count_independent_k5(adj, 5)
    return k5 + ind5

def _is_valid(adj):
    return _total_cost(adj) == 0

def _find_one_k5(adj):
    """Find one K5 in the graph. Returns tuple of 5 vertices or None."""
    n = adj.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                common = np.logical_and(adj[u, :], adj[v, :])
                common[u] = False
                common[v] = False
                indices = np.where(common)[0]
                if len(indices) < 3:
                    continue
                sub_adj = adj[np.ix_(indices, indices)]
                m = len(indices)
                for i in range(m):
                    for j in range(i + 1, m):
                        if sub_adj[i, j]:
                            for k in range(j + 1, m):
                                if sub_adj[i, k] and sub_adj[j, k]:
                                    return (u, v, indices[i], indices[j], indices[k])
    return None

def _find_one_ind5(adj):
    """Find one Independent 5-set in the graph. Returns tuple of 5 vertices or None."""
    n = adj.shape[0]
    comp = ~adj
    np.fill_diagonal(comp, False)
    res = _find_one_k5(comp)
    return res

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Start from Paley graph
    q = 37
    adj_seed = _paley_graph(q)
    
    # Verify seed is valid
    assert _is_valid(adj_seed), "Paley graph on 37 vertices should be (5,5)-free"
    
    best_n = q
    best_adj = adj_seed.copy()
    
    # Try to extend to larger n
    # Strategy: Conflict-Driven Local Search
    # Start from a graph on target_n vertices.
    # If valid, return.
    # Else, find a conflict (K5 or Ind5).
    # Flip an edge in the conflict to destroy it.
    # Repeat.
    
    def _try_find_valid_graph(target_n, time_budget):
        """
        Try to find a valid graph on target_n vertices using conflict-driven search.
        """
        adj_local = np.zeros((target_n, target_n), dtype=bool)
        
        # Initialize with Paley + random extension
        if target_n > q:
            adj_local[:q, :q] = best_adj
            for i in range(q, target_n):
                for j in range(q):
                    if random.random() < 0.5:
                        adj_local[i, j] = adj_local[j, i] = True
            for i in range(q, target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_local[i, j] = adj_local[j, i] = True
        else:
            adj_local = best_adj[:target_n, :target_n].copy()
        
        current_cost = _total_cost(adj_local)
        if current_cost == 0:
            return adj_local
        
        n = target_n
        
        # Conflict-Driven Loop
        while time.time() < start_time + time_budget:
            if current_cost == 0:
                return adj_local
            
            # Find a conflict
            # Randomly choose to look for K5 or Ind5
            if random.random() < 0.5:
                conflict = _find_one_k5(adj_local)
                is_k5 = True
            else:
                conflict = _find_one_ind5(adj_local)
                is_k5 = False
            
            if conflict is None:
                # Should not happen if cost > 0, but just in case
                # Re-evaluate cost
                current_cost = _total_cost(adj_local)
                if current_cost == 0:
                    return adj_local
                continue
            
            # Pick a random edge in the conflict to flip
            # If it's a K5, we must REMOVE an edge.
            # If it's an Ind5, we must ADD an edge.
            # Picking a random edge in the set of 10 edges.
            vertices = list(conflict)
            # Generate all pairs
            edges_in_conflict = [(vertices[i], vertices[j]) for i in range(5) for j in range(i+1, 5)]
            u, v = random.choice(edges_in_conflict)
            
            # Flip the edge
            # For K5: edge exists, so we set to False.
            # For Ind5: edge does NOT exist (in original graph), so we set to True.
            # Note: _find_one_ind5 uses complement. So in adj_local, the edge is False.
            
            if is_k5:
                # Remove edge
                if adj_local[u, v]:
                    # Calculate delta
                    # Removing edge (u,v) from K5 destroys this K5.
                    # It might create Ind5s?
                    # It might destroy other K5s?
                    # We just flip and re-evaluate cost? 
                    # Re-evaluating cost is O(n^3) approx. 
                    # With n=42, it's fast enough for a few thousand steps?
                    # Let's try incremental update if possible, but re-eval is safer.
                    adj_local[u, v] = False
                    adj_local[v, u] = False
                    current_cost = _total_cost(adj_local)
                else:
                    # Edge not present? Should not happen for K5.
                    pass
            else:
                # Add edge
                if not adj_local[u, v]:
                    adj_local[u, v] = True
                    adj_local[v, u] = True
                    current_cost = _total_cost(adj_local)
                else:
                    # Edge present? Should not happen for Ind5.
                    pass
        
        return None
    
    # Try from n=42 down to n=38
    for target_n in [42, 41, 40, 39, 38]:
        remaining = deadline - time.time()
        if remaining < 30:
            break
        # Give more time to 42
        time_budget = min(remaining * 0.8, 100)
        result = _try_find_valid_graph(target_n, time_budget)
        if result is not None:
            if _is_valid(result):
                best_n = target_n
                best_adj = result
                break
    
    n = best_n
    adj_final = best_adj
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_final[i, j]:
                edges.append((i, j))
    
    return n, edges
