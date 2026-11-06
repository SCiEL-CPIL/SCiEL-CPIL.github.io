"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def count_c4s_fast(adj):
    """Count number of C4s efficiently using matrix operations."""
    # common[i,j] = number of common neighbors of i and j
    common = adj @ adj
    np.fill_diagonal(common, 0)
    # Each C4 is counted twice (once for each pair of opposite vertices)
    # For pair (i,j), it contributes C(common[i,j], 2) to the count
    # Total C4s = (1/2) * sum_{i<j} C(common[i,j], 2)
    tri = common[common >= 2]
    if len(tri) == 0:
        return 0
    # C(x,2) = x*(x-1)/2
    c4_sum = np.sum(tri * (tri - 1)) // 2
    return c4_sum // 2

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 31
    target_edges = 91
    
    # Construct initial graph using finite field structure
    # Use GF(5) based construction for a near-C4-free graph
    
    random.seed(12345)
    
    # Strategy: Build graph using modular arithmetic to minimize C4s
    # Partition 31 vertices and use structured connections
    
    # Use the following construction:
    # Label vertices 0-24 as (i,j) in GF(5)xGF(5), 25-29 as "infinity" points, 30 as special
    # Connect based on linear relations to avoid C4s
    
    # For now, use a randomized approach with good initialization
    adj = np.zeros((n, n), dtype=np.int32)
    edges = set()
    
    # Initialize with a random graph
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs[:target_edges]:
        adj[u, v] = 1
        adj[v, u] = 1
        edges.add((u, v))
    
    current_c4 = count_c4s_fast(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edges = set(edges)
    
    # Local search with edge swaps
    edge_list = list(edges)
    non_edge_cache = []
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) not in edges:
                non_edge_cache.append((i,j))
    
    iteration = 0
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        iteration += 1
        
        if len(edge_list) == 0 or len(non_edge_cache) == 0:
            break
        
        # Pick random edge to remove
        e1_idx = random.randint(0, len(edge_list) - 1)
        u1, v1 = edge_list[e1_idx]
        
        # Try several candidate additions
        improved = False
        for _ in range(30):
            idx = random.randint(0, len(non_edge_cache) - 1)
            u2, v2 = non_edge_cache[idx]
            
            # Perform swap
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            edges.discard((u1, v1))
            edges.add((u2, v2))
            
            new_c4 = count_c4s_fast(adj)
            
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = adj.copy()
                best_edges = set(edges)
                no_improve = 0
                improved = True
                break
            else:
                # Revert
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                edges.discard((u2, v2))
                edges.add((u1, v1))
        
        if not improved:
            no_improve += 1
        
        # Update lists
        edge_list = list(edges)
        non_edge_cache = [(i,j) for i in range(n) for j in range(i+1, n) if (i,j) not in edges]
        
        if best_c4 == 0:
            break
    
    # Return best graph
    result_edges = []
    for u, v in best_edges:
        result_edges.append([min(u,v), max(u,v)])
    
    result_edges.sort()
    
    assert len(result_edges) == 91
    assert all(0 <= e[0] <= 30 and 0 <= e[1] <= 30 for e in result_edges)
    assert all(e[0] < e[1] for e in result_edges)
    
    return result_edges
# EVOLVE-BLOCK-END

