import time
import random
import numpy as np

N = 20
DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _popcount(x):
    return bin(x).count('1')

def _can_add_edge_fast(adj, u, v):
    """
    Check if adding edge (u,v) to a K3,3-free graph keeps it K3,3-free.
    adj is a list of int bitmasks.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Candidates for B-side (neighbors of u, excluding v)
    b_mask = nu & ~(1 << v)
    # Candidates for A-side (neighbors of v, excluding u)
    a_mask = nv & ~(1 << u)
    
    # Need at least 2 in each to form K3,3 with u,v
    if _popcount(b_mask) < 2 or _popcount(a_mask) < 2:
        return True
    
    # We need to check if there exist a1, a2 in A and b1, b2 in B
    # such that a1-b1, a1-b2, a2-b1, a2-b2 all exist.
    # This is a K2,2 in the bipartite graph (A, B, E_existing).
    
    # Iterate over pairs in the smaller set to reduce complexity
    # Convert masks to lists of vertices
    a_list = [i for i in range(N) if (a_mask >> i) & 1]
    b_list = [i for i in range(N) if (b_mask >> i) & 1]
    
    if len(a_list) < len(b_list):
        # Iterate pairs in a_list
        for i in range(len(a_list)):
            a1 = a_list[i]
            adj_a1 = adj[a1]
            for j in range(i + 1, len(a_list)):
                a2 = a_list[j]
                adj_a2 = adj[a2]
                # Common neighbors of a1 and a2 within b_mask
                common = adj_a1 & adj_a2 & b_mask
                if _popcount(common) >= 2:
                    return False
    else:
        # Iterate pairs in b_list
        for i in range(len(b_list)):
            b1 = b_list[i]
            adj_b1 = adj[b1]
            for j in range(i + 1, len(b_list)):
                b2 = b_list[j]
                adj_b2 = adj[b2]
                # Common neighbors of b1 and b2 within a_mask
                common = adj_b1 & adj_b2 & a_mask
                if _popcount(common) >= 2:
                    return False
    return True

def _edge_count(adj):
    return sum(_popcount(adj[u]) for u in range(N)) // 2

def _greedy_fill(adj, time_limit):
    """
    Greedily add all possible edges.
    """
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() - start_time > time_limit:
                    return adj
                if (adj[u] >> v) & 1:
                    continue
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
    return adj

def _local_search(adj, time_limit):
    """
    Local search with 1-2 swaps (remove 1, add 2) and 2-2 swaps (remove 2, add 2).
    """
    start_time = time.time()
    current_count = _edge_count(adj)
    best_adj = adj[:]
    best_count = current_count
    
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Precompute list of missing edges for quick access? 
    # It's better to generate on the fly or maintain a set.
    # Given N=20, 190 edges total. Iterating is fast.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1-2 Swap: Remove one edge, try to add two
        # We iterate over existing edges
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() - start_time > time_limit:
                    return best_adj
                if not ((adj[u] >> v) & 1):
                    continue
                
                # Save state
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                added = 0
                new_edges = []
                
                # Try to add missing edges
                for u2 in range(N):
                    for v2 in range(u2 + 1, N):
                        if (adj[u2] >> v2) & 1:
                            continue
                        if _can_add_edge_fast(adj, u2, v2):
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                            added += 1
                            new_edges.append((u2, v2))
                
                if added >= 2:
                    # Net gain is added - 1. If added >= 2, gain >= 1.
                    # Actually, if added == 1, gain 0. If added > 1, gain > 0.
                    # We want to strictly improve or maintain? Strictly improve is safer for convergence.
                    # But if added == 1, we might be stuck. 
                    # Let's accept if added > 1 (net gain >= 1).
                    # If added == 1, we revert.
                    if added > 1:
                        current_count += (added - 1)
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                    else:
                        # Revert
                        for eu, ev in new_edges:
                            adj[eu] &= ~(1 << ev)
                            adj[ev] &= ~(1 << eu)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    # Revert
                    for eu, ev in new_edges:
                        adj[eu] &= ~(1 << ev)
                        adj[ev] &= ~(1 << eu)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        if not improved:
            break
            
    return best_adj

def _random_start(density):
    """Generate a random graph with given edge probability, filtered for K3,3."""
    adj = [0] * N
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < density:
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_adj = [0] * N
    best_count = 0
    
    # Strategy 1: High density random starts + Greedy + Local Search
    densities = [0.6, 0.7, 0.8, 0.5, 0.4]
    
    # Shuffle densities for variety
    random.shuffle(densities)
    
    for density in densities:
        if time.time() > deadline - 20:
            break
            
        adj = _random_start(density)
        
        # Greedy fill
        remaining = deadline - time.time() - 5
        if remaining < 1:
            break
        adj = _greedy_fill(adj, min(remaining * 0.4, 5.0))
        
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
        
        # Local search
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        adj = _local_search(adj, remaining * 0.8)
        
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
            
    # If we still have time, do more local search on the best found
    remaining = deadline - time.time() - 1
    if remaining > 2:
        adj = best_adj[:]
        adj = _local_search(adj, remaining)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
