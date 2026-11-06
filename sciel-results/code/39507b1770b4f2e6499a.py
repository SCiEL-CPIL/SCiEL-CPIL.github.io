"""
Program to construct a K3,3-free graph on 20 vertices maximizing edges.

Approach:
1. Use spectral heuristic to initialize a dense graph
2. Apply local search to remove edges that create K3,3 subgraphs
3. The key insight is that a K3,3 exists iff there exist 3 vertices with 3+ common neighbours
"""
import numpy as np
from itertools import combinations
import random

def construct():
    n = 20
    random.seed(42)
    np.random.seed(42)
    
    # Spectral heuristic initialization
    # Create a random symmetric matrix
    A_rand = np.random.randn(n, n)
    A_rand = (A_rand + A_rand.T) / 2
    
    # Get top eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(A_rand)
    top_eigenvector = eigenvectors[:, -1]  # Largest eigenvalue's eigenvector
    
    # Use magnitude of eigenvector components to create initial adjacency
    # Threshold at mean to create a balanced graph
    vec_abs = np.abs(top_eigenvector)
    threshold = np.mean(vec_abs)
    
    # Create initial adjacency matrix
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if vec_abs[i] > threshold and vec_abs[j] > threshold:
                adj[i, j] = adj[j, i] = 1
    
    # Convert to edge list
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    edges.append([i, j])
        return edges
    
    def has_k33(adj, u, v, w):
        """Check if vertices u, v, w have 3 or more common neighbours."""
        common = set(adj[u].nonzero()[0]) & set(adj[v].nonzero()[0]) & set(adj[w].nonzero()[0])
        return len(common) >= 3
    
    def count_k33_violations(adj):
        """Count the number of triples with 3+ common neighbours."""
        violations = 0
        for u, v, w in combinations(range(n), 3):
            if has_k33(adj, u, v, w):
                violations += 1
        return violations
    
    # Local search: remove edges that are part of K3,3 subgraphs
    best_edges = adj_to_edges(adj)
    best_violations = count_k33_violations(adj)
    
    for iteration in range(500):
        if best_violations == 0:
            break
            
        # Find all edges that are in at least one K3,3
        edges_in_k33 = set()
        for u, v, w in combinations(range(n), 3):
            if has_k33(adj, u, v, w):
                common = set(adj[u].nonzero()[0]) & set(adj[v].nonzero()[0]) & set(adj[w].nonzero()[0])
                for x in common:
                    for y in common:
                        if x < y:
                            edges_in_k33.add((x, y))
        
        # Remove an edge that is in the most K3,3 subgraphs
        if edges_in_k33:
            # Pick a random edge from the violating edges
            edge_list = list(edges_in_k33)
            edge = random.choice(edge_list)
            u, v = edge
            if adj[u, v] == 1:
                adj[u, v] = adj[v, u] = 0
                new_violations = count_k33_violations(adj)
                if new_violations <= best_violations:
                    best_violations = new_violations
                    best_edges = adj_to_edges(adj)
                else:
                    adj[u, v] = adj[v, u] = 1  # Undo
    
    # If still has violations, do a more aggressive cleanup
    while best_violations > 0:
        adj = np.array(best_edges_to_matrix(best_edges, n))
        violations_list = []
        for u, v, w in combinations(range(n), 3):
            if has_k33(adj, u, v, w):
                common = set(adj[u].nonzero()[0]) & set(adj[v].nonzero()[0]) & set(adj[w].nonzero()[0])
                for x in common:
                    for y in common:
                        if x < y:
                            violations_list.append((x, y))
        
        if not violations_list:
            break
        
        # Remove all violating edges
        for x, y in set(violations_list):
            if adj[x, y] == 1:
                adj[x, y] = adj[y, x] = 0
        
        best_edges = adj_to_edges(adj)
        best_violations = count_k33_violations(adj)
    
    return best_edges

def best_edges_to_matrix(edges, n):
    """Convert edge list to adjacency matrix."""
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1
    return adj
