"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_55_free_fast(n, adj):
    """Check if graph is (5,5)-free using bitmask adjacency with early termination."""
    all_mask = (1 << n) - 1
    non_adj = [all_mask ^ adj[i] for i in range(n)]
    
    # Check for K5 (clique of size 5)
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not (adj[v0] >> v1) & 1:
                continue
            for v2 in range(v1 + 1, n):
                if not ((adj[v0] >> v2) & 1) and not ((adj[v1] >> v2) & 1):
                    continue
                for v3 in range(v2 + 1, n):
                    if not ((adj[v0] >> v3) & 1) and not ((adj[v1] >> v3) & 1) and not ((adj[v2] >> v3) & 1):
                        continue
                    for v4 in range(v3 + 1, n):
                        if ((adj[v0] >> v4) & 1) and ((adj[v1] >> v4) & 1) and ((adj[v2] >> v4) & 1) and ((adj[v3] >> v4) & 1):
                            return False
    # Check for independent set of size 5
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if (adj[v0] >> v1) & 1:
                continue
            for v2 in range(v1 + 1, n):
                if ((adj[v0] >> v2) & 1) or ((adj[v1] >> v2) & 1):
                    continue
                for v3 in range(v2 + 1, n):
                    if ((adj[v0] >> v3) & 1) or ((adj[v1] >> v3) & 1) or ((adj[v2] >> v3) & 1):
                        continue
                    for v4 in range(v3 + 1, n):
                        if ((adj[v0] >> v4) & 1) or ((adj[v1] >> v4) & 1) or ((adj[v2] >> v4) & 1) or ((adj[v3] >> v4) & 1):
                            return False
    return True

def _check_new_vertex_valid(n, adj, new_vertex_mask):
    """
    Check if adding a new vertex (index n) with the given adjacency mask
    keeps the graph (5,5)-free. Only checks tuples that include the new vertex.
    This is much faster than full verification.
    """
    # The new vertex is at index n
    # Check for K5 that includes the new vertex
    # A K5 including new vertex means we need 4 other vertices that form a K4
    # and are all adjacent to the new vertex
    neighbors = []
    for i in range(n):
        if (new_vertex_mask >> i) & 1:
            neighbors.append(i)
    
    # Check all 4-subsets of neighbors for being a K4
    for combo in combinations(neighbors, 4):
        v0, v1, v2, v3 = combo
        # Check if these 4 form a clique
        if ((adj[v0] >> v1) & 1) and ((adj[v0] >> v2) & 1) and ((adj[v0] >> v3) & 1) and \
           ((adj[v1] >> v2) & 1) and ((adj[v1] >> v3) & 1) and \
           ((adj[v2] >> v3) & 1):
            return False  # Found K5 with new vertex
    
    # Check for independent set of size 5 that includes the new vertex
    # Non-neighbors of new vertex
    non_neighbors = []
    for i in range(n):
        if not ((new_vertex_mask >> i) & 1):
            non_neighbors.append(i)
    
    # Check all 4-subsets of non-neighbors for being an independent 4-set
    for combo in combinations(non_neighbors, 4):
        v0, v1, v2, v3 = combo
        # Check if these 4 are mutually non-adjacent
        if not ((adj[v0] >> v1) & 1) and not ((adj[v0] >> v2) & 1) and not ((adj[v0] >> v3) & 1) and \
           not ((adj[v1] >> v2) & 1) and not ((adj[v1] >> v3) & 1) and \
           not ((adj[v2] >> v3) & 1):
            return False  # Found independent 5-set with new vertex
    
    return True

def _paley_graph(q):
    """Construct Paley graph of prime order q (q ≡ 1 mod 4). Returns (n, adj, edges)."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    return q, adj, edges

def _try_extend_graph(adj, n, deadline, max_extensions=20):
    """Try to extend graph by adding one vertex at a time with smarter heuristics."""
    best_n = n
    best_adj = list(adj)
    rng = random.Random(12345)
    
    for ext_round in range(max_extensions):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try many random patterns, biased toward different densities
        for attempt in range(200):
            if time.time() > deadline - 15:
                break
            
            # Vary the connection probability
            # For a (5,5)-free graph, we want roughly balanced adjacency
            p = rng.uniform(0.3, 0.7)
            
            new_adj_mask = 0
            for i in range(n):
                if rng.random() < p:
                    new_adj_mask |= (1 << i)
            
            # Quick check: count neighbors and non-neighbors
            # If too many neighbors, likely creates K5; if too few, likely creates independent set
            num_neighbors = bin(new_adj_mask).count('1')
            num_non_neighbors = n - num_neighbors
            
            # Heuristic pruning: if neighbors > n*0.6 or non_neighbors > n*0.6, skip
            if num_neighbors > int(n * 0.65) or num_non_neighbors > int(n * 0.65):
                continue
            
            if _check_new_vertex_valid(n, best_adj, new_adj_mask):
                # Update adjacency
                new_n = n + 1
                new_adj = list(best_adj) + [new_adj_mask]
                for i in range(n):
                    if (new_adj_mask >> i) & 1:
                        new_adj[i] |= (1 << n)
                
                best_n = new_n
                best_adj = new_adj
                n = new_n
                found = True
                break
        
        if not found:
            # Try some structured patterns before giving up
            for attempt in range(20):
                if time.time() > deadline - 15:
                    break
                
                # Pattern: connect to roughly half, but try different biases
                bias = 0.5 + rng.uniform(-0.15, 0.15)
                new_adj_mask = 0
                for i in range(n):
                    if rng.random() < bias:
                        new_adj_mask |= (1 << i)
                
                num_neighbors = bin(new_adj_mask).count('1')
                num_non_neighbors = n - num_neighbors
                if num_neighbors > int(n * 0.6) or num_non_neighbors > int(n * 0.6):
                    continue
                
                if _check_new_vertex_valid(n, best_adj, new_adj_mask):
                    new_n = n + 1
                    new_adj = list(best_adj) + [new_adj_mask]
                    for i in range(n):
                        if (new_adj_mask >> i) & 1:
                            new_adj[i] |= (1 << n)
                    
                    best_n = new_n
                    best_adj = new_adj
                    n = new_n
                    found = True
                    break
        
        if not found:
            break
    
    return best_n, best_adj

def _local_search(adj, n, deadline, iterations=2000):
    """Local search: try flipping edges to see if we can maintain (5,5)-freeness."""
    best_adj = list(adj)
    best_n = n
    rng = random.Random(42)
    
    for _ in range(iterations):
        if time.time() > deadline - 15:
            break
        
        # Pick a random edge to flip
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        new_adj = list(best_adj)
        # Flip edge (i, j)
        if (best_adj[i] >> j) & 1:
            new_adj[i] &= ~(1 << j)
            new_adj[j] &= ~(1 << i)
        else:
            new_adj[i] |= (1 << j)
            new_adj[j] |= (1 << i)
        
        if _is_55_free_fast(n, new_adj):
            best_adj = new_adj
    
    return best_n, best_adj

def construct():
    deadline = time.time() + 280  # 280s safety margin (killed at 330s)
    
    # Start with Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    
    best_n = q
    best_adj = list(adj)
    
    # Try to extend the graph with smarter heuristics
    extended_n, extended_adj = _try_extend_graph(adj, q, deadline, max_extensions=20)
    if extended_n > best_n:
        best_n = extended_n
        best_adj = extended_adj
    
    # Try local search on the extended graph
    ls_n, ls_adj = _local_search(best_adj, best_n, deadline, iterations=3000)
    if ls_n > best_n:
        best_n = ls_n
        best_adj = ls_adj
    
    # Convert to edge list
    best_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                best_edges.append((i, j))
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
