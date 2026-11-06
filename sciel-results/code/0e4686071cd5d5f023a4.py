"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _has_k5(adj, n):
    """Check if graph with adjacency matrix adj on n vertices has a K5."""
    for clique in combinations(range(n), 5):
        for i in range(5):
            for j in range(i+1, 5):
                if not adj[clique[i]][clique[j]]:
                    break
            else:
                continue
            break
        else:
            return True
    return False

def _has_indep5(adj, n):
    """Check if graph with adjacency matrix adj on n vertices has an independent set of size 5."""
    for indep in combinations(range(n), 5):
        for i in range(5):
            for j in range(i+1, 5):
                if adj[indep[i]][indep[j]]:
                    break
            else:
                continue
            break
        else:
            return True
    return False

def _is_55_free(adj, n):
    """Check if graph is (5,5)-free: no K5 and no independent set of size 5."""
    if _has_k5(adj, n):
        return False
    if _has_indep5(adj, n):
        return False
    return True

def _try_add_vertex(adj, n, new_conn):
    """Try adding a new vertex with connection pattern new_conn (bitmask of which old vertices it connects to).
    Returns True if the resulting graph is (5,5)-free."""
    # Build new adjacency
    new_n = n + 1
    new_adj = [[False] * new_n for _ in range(new_n)]
    for i in range(n):
        for j in range(n):
            new_adj[i][j] = adj[i][j]
    for i in range(n):
        new_adj[i][new_n-1] = bool(new_conn & (1 << i))
        new_adj[new_n-1][i] = new_adj[i][new_n-1]
    return _is_55_free(new_adj, new_n)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    # Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[False] * q for _ in range(q)]
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i][j] = True
                adj[j][i] = True
    
    current_n = q
    current_adj = adj
    best_n = q
    best_adj = adj
    
    # Try to add vertices one at a time
    for target_n in range(current_n + 1, 47):
        if time.time() > deadline:
            break
        
        # Try all possible connection patterns for the new vertex
        # There are 2^n possible patterns, but we can be smarter
        # A vertex can connect to at most 3 of any 4 existing vertices (else K5)
        # And at most 3 of any 4 existing vertices must be non-connected (else indep5)
        
        found = False
        # Use random sampling of connection patterns
        import random
        attempts = 0
        max_attempts = 50000
        
        while attempts < max_attempts and not found:
            if time.time() > deadline:
                break
            
            # Random connection pattern
            new_conn = random.getrandbits(current_n)
            
            # Quick check: count connections
            conn_count = bin(new_conn).count('1')
            # If connected to too many, likely creates K5
            # If not connected to enough, likely creates independent set
            # Heuristic: balance connections
            if conn_count > current_n * 0.6 or conn_count < current_n * 0.4:
                attempts += 1
                continue
            
            if _try_add_vertex(current_adj, current_n, new_conn):
                # Build the new adjacency
                new_n = current_n + 1
                new_adj = [[False] * new_n for _ in range(new_n)]
                for i in range(current_n):
                    for j in range(current_n):
                        new_adj[i][j] = current_adj[i][j]
                for i in range(current_n):
                    new_adj[i][new_n-1] = bool(new_conn & (1 << i))
                    new_adj[new_n-1][i] = new_adj[i][new_n-1]
                
                current_adj = new_adj
                current_n = new_n
                found = True
                break
            
            attempts += 1
        
        if not found:
            break
    
    # Return the best graph found
    n = current_n
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if current_adj[i][j]:
                edges.append((i, j))
    
    return n, edges
# EVOLVE-BLOCK-END
