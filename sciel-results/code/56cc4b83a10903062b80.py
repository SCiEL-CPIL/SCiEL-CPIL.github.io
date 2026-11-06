import time
import random
import itertools
import sys

def has_k5(adj, n):
    """Check if there is a K5 in the graph."""
    # Use a fast check: for each 5-subset, check if it's a clique.
    # For n=42, C(42,5) is ~850k. This is fast enough for a few iterations.
    for combo in itertools.combinations(range(n), 5):
        is_clique = True
        for i in range(5):
            for j in range(i+1, 5):
                if not adj[combo[i]][combo[j]]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            return combo
    return None

def has_independent_5(adj, n):
    """Check if there is an independent set of size 5."""
    # Independent set in G is a clique in G_complement.
    # We can just check if all pairs are non-edges.
    for combo in itertools.combinations(range(n), 5):
        is_indep = True
        for i in range(5):
            for j in range(i+1, 5):
                if adj[combo[i]][combo[j]]:
                    is_indep = False
                    break
            if not is_indep:
                break
        if is_indep:
            return combo
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Fallback: 37-vertex Paley graph
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges_37 = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Try to find a 42-vertex graph
    n_target = 42
    found_valid = False
    best_n = q
    best_edges = edges_37
    
    # Initialize a random graph with p=0.5
    # Or start with a known dense graph?
    # Let's try a few random restarts.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize adjacency matrix
        adj = [[0]*n_target for _ in range(n_target)]
        for i in range(n_target):
            for j in range(i+1, n_target):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Local search
        # We want to eliminate K5s and Independent 5-sets.
        # Strategy: 
        # 1. Find a K5. If found, remove a random edge from it.
        # 2. Find an Independent 5-set. If found, add a random edge within it.
        # 3. If neither, we are valid.
        
        # To speed up, we can cache violations?
        # For n=42, the check is ~1M operations. 1000 iterations = 1B ops. Might be slow in Python.
        # Optimization: Only check if the graph changed?
        
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Check for K5
            k5 = has_k5(adj, n_target)
            if k5:
                # Remove a random edge from the K5
                u, v = random.sample(k5, 2)
                adj[u][v] = 0
                adj[v][u] = 0
                continue
                
            # Check for Independent 5-set
            i5 = has_independent_5(adj, n_target)
            if i5:
                # Add a random edge within the independent set
                u, v = random.sample(i5, 2)
                if not adj[u][v]:
                    adj[u][v] = 1
                    adj[v][u] = 1
                else:
                    # This shouldn't happen if it's an independent set
                    pass
                continue
            
            # If neither K5 nor Independent 5-set, we have a valid graph
            found_valid = True
            break
            
        if found_valid:
            # Construct edge list
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if adj[i][j]:
                        edges.append((i, j))
            return n_target, edges

    # If no 42-vertex graph found, return 37-vertex
    return best_n, best_edges
