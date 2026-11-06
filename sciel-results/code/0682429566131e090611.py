"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def _count_c4s(edges, n=31):
    """Count the number of 4-cycles in a graph given as list of edges."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    c4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if j in adj[i]:
                common = adj[i] & adj[j]
                c4_count += len(common)
    return c4_count // 2

def _greedy_c4_free(n=31, target_edges=90, seed=42, deadline=None):
    """Greedy construction of a C4-free graph with target_edges edges."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    common = [[0]*n for _ in range(n)]  # common[i][j] = number of common neighbors
    edges = []
    
    # Precompute all possible edges and shuffle
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if len(edges) >= target_edges:
            break
        if deadline is not None and time.time() > deadline:
            break
        # Check if adding (u,v) creates a C4
        if common[u][v] == 0:
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            # Update common neighbor counts
            for w in adj[u]:
                if w != v:
                    common[w][v] += 1
                    common[v][w] += 1
            for w in adj[v]:
                if w != u:
                    common[u][w] += 1
                    common[w][u] += 1
            common[u][v] = len(adj[u] & adj[v])
            common[v][u] = common[u][v]
    
    return edges

def construct():
    deadline = time.time() + 120.0
    best_edges = None
    best_c4 = float('inf')
    
    # Try multiple random seeds to find a C4-free graph with 90 edges
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        edges90 = _greedy_c4_free(31, 90, seed, deadline - 5)
        if len(edges90) < 90:
            continue
        
        # Verify it's actually C4-free
        c4_count = _count_c4s(edges90)
        if c4_count != 0:
            continue
        
        # Add one edge to get 91 edges, choosing the edge that minimizes C4s
        adj = [set() for _ in range(31)]
        for u, v in edges90:
            adj[u].add(v)
            adj[v].add(u)
        
        best_add = None
        best_add_c4 = float('inf')
        for i in range(31):
            for j in range(i+1, 31):
                if j not in adj[i]:
                    common = len(adj[i] & adj[j])
                    if common < best_add_c4:
                        best_add_c4 = common
                        best_add = (i, j)
        
        if best_add is None:
            continue
        
        new_edges = edges90 + [best_add]
        total_c4 = _count_c4s(new_edges)
        
        if total_c4 < best_c4:
            best_c4 = total_c4
            best_edges = new_edges
            if best_c4 <= 1:
                break
    
    if best_edges is None:
        # Fallback: use the first 91 edges (shouldn't happen)
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
