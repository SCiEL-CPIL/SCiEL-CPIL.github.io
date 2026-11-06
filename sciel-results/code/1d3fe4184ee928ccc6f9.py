"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_k4(adj, n):
    """Check if graph has a K4. adj is list of sets."""
    for a in range(n):
        for b in adj[a]:
            if b <= a: continue
            common_ab = adj[a] & adj[b]
            for c in common_ab:
                if c <= b: continue
                common_abc = common_ab & adj[c]
                for d in common_abc:
                    if d > c:
                        return True
    return False

def _has_independent_6(adj, n):
    """Check if graph has an independent set of size 6."""
    # Use complement: independent set in G = clique in complement
    # Complement adjacency: i~j in complement iff i!=j and j not in adj[i]
    for a in range(n):
        non_neigh_a = set(range(n)) - {a} - adj[a]
        for b in non_neigh_a:
            non_neigh_ab = non_neigh_a & (set(range(n)) - {b} - adj[b])
            for c in non_neigh_ab:
                non_neigh_abc = non_neigh_ab & (set(range(n)) - {c} - adj[c])
                for d in non_neigh_abc:
                    non_neigh_abcd = non_neigh_abc & (set(range(n)) - {d} - adj[d])
                    for e in non_neigh_abcd:
                        if len(non_neigh_abcd & (set(range(n)) - {e} - adj[e])) >= 1:
                            return True
    return False

def _is_valid(adj, n):
    return not _has_k4(adj, n) and not _has_independent_6(adj, n)

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in adj[i]:
            if j > i:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s budget, 10s safety margin
    best_n = 17
    best_edges = None
    
    # First, try the known Schläfli graph construction (27 vertices)
    # Schläfli graph: vertices are (x,y,z) in Z_3^3
    # Two vertices adjacent if they agree in exactly 0 or 2 coordinates
    # Wait, let me use the standard: adjacent if Hamming distance is 1 or 2 in Z_3^3?
    # Actually: in Z_3^3, two points are adjacent if they differ in exactly 1 or 2 coordinates
    # Let me verify this gives the Schläfli graph parameters.
    
    # Standard Schläfli: vertices Z_3^3, adjacent if coordinate-wise difference is (0,0,0) excluded,
    # and they are adjacent if the number of coordinates where they differ is 1 or 2.
    # Degree: for a fixed vertex, number of vectors at Hamming distance 1: 3*2=6
    # Hamming distance 2: C(3,2)*2*2=12
    # Total degree = 18. But Schläfli has degree 16. So this is wrong.
    
    # Correct Schläfli: adjacent if they agree in exactly 1 coordinate (i.e., differ in 2)
    # or agree in exactly 0 coordinates (differ in 3)? 
    # Degree: agree in 1 coord: C(3,1)*2*2=12, agree in 0: 2*2*2=8, total=20. No.
    
    # Let me just use the random approach with verification.
    
    random.seed(42)
    
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random graphs
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random graph with p=0.5
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i].add(j)
                        adj[j].add(i)
            
            if _is_valid(adj, n):
                best_n = n
                best_edges = _edges_from_adj(adj, n)
                return {'n': n, 'edges': best_edges}
    
    # Fallback: return the Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
# EVOLVE-BLOCK-END
