import time
import random
import numpy as np
from itertools import combinations

def _is_valid_graph(adj_matrix):
    """
    Check if the graph has no regular induced subgraph on 6 or more vertices.
    Uses vectorized NumPy operations.
    """
    n = adj_matrix.shape[0]
    
    # Check all subsets of size 6 to n
    for size in range(6, n + 1):
        # Generate all combinations of size `size`
        for combo in combinations(range(n), size):
            # Extract the induced subgraph
            sub_adj = adj_matrix[np.ix_(combo, combo)]
            
            # Compute degrees in the induced subgraph
            degrees = sub_adj.sum(axis=1)
            
            # Check if all degrees are equal (regular)
            if np.all(degrees == degrees[0]):
                return False
    
    return True

def _count_violations(adj_matrix):
    """
    Count the number of regular induced subgraphs of size >= 6.
    Returns (count, list_of_violating_subsets) for the first few violations.
    """
    n = adj_matrix.shape[0]
    violations = []
    
    for size in range(6, n + 1):
        for combo in combinations(range(n), size):
            sub_adj = adj_matrix[np.ix_(combo, combo)]
            degrees = sub_adj.sum(axis=1)
            if np.all(degrees == degrees[0]):
                violations.append(combo)
                if len(violations) >= 100:
                    return len(violations), violations
    
    return len(violations), violations

def _edges_from_adj_matrix(adj_matrix):
    """Convert adjacency matrix to edge list."""
    n = adj_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] == 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to find valid graphs for n = 22, 21, 20, 19
    for target_n in [22, 21, 20, 19]:
        if time.time() > deadline - 10:
            break
        
        n = target_n
        found_valid = False
        
        # Strategy 1: Random graphs with controlled edge density
        # Strategy 2: Structured graphs based on degree sequences
        
        # Use multiple random restarts
        max_restarts = 50
        restarts_attempted = 0
        
        while restarts_attempted < max_restarts and time.time() < deadline - 10:
            restarts_attempted += 1
            
            # Generate a random graph
            # Use different edge probabilities to explore the space
            p = 0.5
            
            # Create adjacency matrix
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
            
            # Check validity
            if _is_valid_graph(adj):
                edges = _edges_from_adj_matrix(adj)
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found_valid = True
                break
            
            # If not valid, try local search: flip edges to break regular subgraphs
            # First, find some violating subgraphs
            _, violations = _count_violations(adj)
            
            if not violations:
                # Should have been caught above, but just in case
                continue
            
            # Try to fix violations by flipping edges
            for _ in range(100):  # Local search iterations
                if time.time() > deadline - 10:
                    break
                
                # Pick a random violation
                combo = random.choice(violations)
                
                # Pick two vertices from the violation to flip their edge
                u, v = random.sample(combo, 2)
                
                # Flip the edge
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
                
                # Check if this fixed the violation (and didn't create new ones)
                # For efficiency, just check the specific violation
                sub_adj = adj[np.ix_(combo, combo)]
                degrees = sub_adj.sum(axis=1)
                if not np.all(degrees == degrees[0]):
                    # This violation is fixed, check full validity
                    if _is_valid_graph(adj):
                        edges = _edges_from_adj_matrix(adj)
                        if n > best_n:
                            best_n = n
                            best_edges = edges
                        found_valid = True
                        break
                
                # Update violations list periodically
                if _ % 20 == 0:
                    _, violations = _count_violations(adj)
                    if not violations:
                        # Should have been caught, but just in case
                        if _is_valid_graph(adj):
                            edges = _edges_from_adj_matrix(adj)
                            if n > best_n:
                                best_n = n
                                best_edges = edges
                            found_valid = True
                        break
            
            if found_valid:
                break
        
        if found_valid:
            # Found a valid graph for this n, try larger n
            continue
    
    # Fallback
    if best_n <= 5:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
