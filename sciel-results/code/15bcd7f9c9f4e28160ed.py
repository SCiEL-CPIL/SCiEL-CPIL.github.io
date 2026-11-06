import time
import sys

def construct():
    # Use Paley graph P(17) on vertices 0..16
    # Add vertex 17 connected to an independent set in P(17)
    # This ensures K4-free and minimizes max triangle-free induced subgraph
    
    n = 18
    edges = []
    
    # Quadratic residues mod 17
    residues = set([i*i % 17 for i in range(17)])
    residues.discard(0)
    
    # Build P(17)
    for u in range(17):
        for v in range(u+1, 17):
            if (v - u) % 17 in residues:
                edges.append([u, v])
                
    # Find an independent set in P(17)
    # Heuristic: Greedy or known max independent set
    # P(17) has alpha=4.
    # Let's find one.
    adj = [[False]*17 for _ in range(17)]
    for e in edges:
        adj[e[0]][e[1]] = True
        adj[e[1]][e[0]] = True
        
    # Greedy independent set
    indep = []
    remaining = list(range(17))
    while remaining:
        # Pick vertex with min degree in remaining
        min_deg = float('inf')
        min_v = -1
        for v in remaining:
            deg = sum(1 for u in remaining if adj[v][u])
            if deg < min_deg:
                min_deg = deg
                min_v = v
        indep.append(min_v)
        remaining.remove(min_v)
        # Remove neighbors
        to_remove = [u for u in remaining if adj[min_v][u]]
        for u in to_remove:
            remaining.remove(u)
            
    # Connect 17 to the independent set
    for v in indep:
        edges.append([17, v])
        
    return edges
