"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score_graph(n, edges):
    """Compute the max degree sum over all triangles."""
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
    
    deg = [sum(adj[i]) for i in range(n)]
    
    best = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                continue
            for w in range(v + 1, n):
                if adj[u][w] and adj[v][w]:
                    s = deg[u] + deg[v] + deg[w]
                    if s > best:
                        best = s
    return best

def construct():
    n = 16
    deadline = time.time() + 120
    
    # Start with K_{8,8} + one intra-part edge
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((min(u,v), max(u,v)))
    edges.add((0, 1))
    
    best_edges = set(edges)
    best_score = _score_graph(n, list(best_edges))
    
    # Local search: try edge swaps (remove one edge, add another)
    all_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.add((i, j))
    
    random.seed(42)
    max_iters = 100000
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        cur_edges = set(best_edges)
        cur_score = best_score
        
        # Try a random swap: remove a random edge, add a random non-edge
        removed = random.choice(list(cur_edges))
        non_edges = all_pairs - cur_edges
        if not non_edges:
            continue
        added = random.choice(list(non_edges))
        
        new_edges = cur_edges - {removed} | {added}
        new_score = _score_graph(n, list(new_edges))
        
        if new_score < best_score:
            best_score = new_score
            best_edges = new_edges
    
    # Convert to list format
    edge_list = [[u, v] for u, v in sorted(best_edges)]
    return {'n': n, 'edges': edge_list}
# EVOLVE-BLOCK-END
