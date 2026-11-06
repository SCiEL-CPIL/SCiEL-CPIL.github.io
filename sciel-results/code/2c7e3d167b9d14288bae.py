"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_55_free_fast(n, adj):
    """Check if graph is (5,5)-free using bitmask adjacency with early termination."""
    # Precompute non-adjacency masks for faster independent set checks
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

def _try_extend_graph(adj, n, deadline, max_extensions=10):
    """Try to extend graph by adding one vertex at a time."""
    best_n = n
    best_adj = list(adj)
    
    for _ in range(max_extensions):
        if time.time() > deadline - 10:
            break
        
        # Try adding a new vertex with various adjacency patterns
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random adjacency pattern for new vertex
            new_adj_mask = 0
            for i in range(n):
                if random.random() < 0.5:
                    new_adj_mask |= (1 << i)
            
            # Check if adding this vertex keeps graph (5,5)-free
            # New vertex index is n
            new_n = n + 1
            new_adj = list(best_adj) + [new_adj_mask]
            for i in range(n):
                if (new_adj_mask >> i) & 1:
                    new_adj[i] |= (1 << n)
            
            if _is_55_free_fast(new_n, new_adj):
                best_n = new_n
                best_adj = new_adj
                n = new_n
                found = True
                break
        
        if not found:
            break
    
    return best_n, best_adj

def _local_search(adj, n, deadline, iterations=1000):
    """Local search: try flipping edges to see if we can maintain (5,5)-freeness."""
    best_adj = list(adj)
    best_n = n
    rng = random.Random(42)
    
    for _ in range(iterations):
        if time.time() > deadline - 10:
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
    deadline = time.time() + 270  # 270s safety margin (killed at 330s)
    
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
    
    # Try to extend the graph
    extended_n, extended_adj = _try_extend_graph(adj, q, deadline, max_extensions=15)
    if extended_n > best_n:
        best_n = extended_n
        best_adj = extended_adj
    
    # Try local search on the extended graph
    ls_n, ls_adj = _local_search(best_adj, best_n, deadline, iterations=500)
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
