import time
import random
import numpy as np

N = 20
DEADLINE = 115.0

def _popcount(x):
    return bin(x).count('1')

def _can_add_edge(adj, u, v):
    """
    Check if adding edge (u, v) to graph with adjacency bitmasks 'adj' 
    preserves K3,3-free property.
    """
    # Neighbors of u excluding v
    nu = adj[u] & ~(1 << v)
    # Neighbors of v excluding u
    nv = adj[v] & ~(1 << u)
    
    # If u has < 2 neighbors or v has < 2 neighbors, no K3,3 can be formed
    if _popcount(nu) < 2 or _popcount(nv) < 2:
        return True
        
    # Get lists of neighbors
    # Optimization: We only need to check pairs in the smaller set?
    # No, we need pairs from one side and check common neighbors in the other.
    # Let A be neighbors of v (excluding u)
    # Let B be neighbors of u (excluding v)
    # We need to ensure there are no a1, a2 in A and b1, b2 in B
    # such that a1-b1, a1-b2, a2-b1, a2-b2 are all edges.
    
    # This is equivalent to:
    # For any pair a1, a2 in A, the set of common neighbors of a1 and a2 
    # within B must have size < 2.
    
    # Let's iterate over pairs in A
    # If |A| is large, this is slow. But N=20, so max 18 neighbors.
    # Pairs: 18*17/2 = 153.
    # For each pair, check popcount(adj[a1] & adj[a2] & B) < 2.
    
    # To speed up, we can iterate over the smaller of A and B.
    # Let's assume A is neighbors of v.
    a_list = [i for i in range(N) if (nv >> i) & 1]
    b_mask = nu
    
    for i in range(len(a_list)):
        a1 = a_list[i]
        for j in range(i+1, len(a_list)):
            a2 = a_list[j]
            # Common neighbors of a1 and a2 that are also in B
            common = adj[a1] & adj[a2] & b_mask
            if _popcount(common) >= 2:
                return False
    return True

def _edge_count(adj):
    count = 0
    for u in range(N):
        count += _popcount(adj[u])
    return count // 2

def _greedy_fill(adj, deadline):
    """
    Greedily add edges in a random order until no more can be added or deadline.
    """
    edges_to_try = [(u, v) for u in range(N) for v in range(u+1, N)]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if time.time() > deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _local_search(adj, deadline):
    """
    Local search with Add and 1-2 Swap moves.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any missing edge
        for u in range(N):
            for v in range(u+1, N):
                if time.time() > deadline:
                    return adj
                if (adj[u] >> v) & 1:
                    continue
                if _can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        if time.time() > deadline:
            return adj

        # 2. Try 1-2 Swaps (Remove 1 edge, Add 2 edges)
        # We need to be careful to restore state if no gain.
        for u in range(N):
            for v in range(u+1, N):
                if time.time() > deadline:
                    return adj
                if not ((adj[u] >> v) & 1):
                    continue
                
                # Remove edge (u, v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                added = 0
                new_edges = []
                
                # Try to add all possible edges
                for u2 in range(N):
                    for v2 in range(u2+1, N):
                        if time.time() > deadline:
                            break
                        if (adj[u2] >> v2) & 1:
                            continue
                        if _can_add_edge(adj, u2, v2):
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                            added += 1
                            new_edges.append((u2, v2))
                
                if time.time() > deadline:
                    break
                    
                if added > 1:
                    # Net gain
                    improved = True
                    # Keep state
                else:
                    # Restore state
                    for eu, ev in new_edges:
                        adj[eu] &= ~(1 << ev)
                        adj[ev] &= ~(1 << eu)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                if time.time() > deadline:
                    return adj
    return adj

def _random_start(density):
    adj = [0] * N
    for u in range(N):
        for v in range(u+1, N):
            if random.random() < density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _k218_start():
    adj = [0] * N
    for v in range(2, N):
        adj[0] |= (1 << v)
        adj[v] |= (1 << 0)
        adj[1] |= (1 << v)
        adj[v] |= (1 << 1)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_adj = [0] * N
    best_count = 0
    
    # Strategy 1: K2,18 base
    adj = _k218_start()
    adj = _greedy_fill(adj, deadline - 5)
    adj = _local_search(adj, deadline - 2)
    count = _edge_count(adj)
    if count > best_count:
        best_adj = adj[:]
        best_count = count
    
    if time.time() > deadline - 10:
        return _adj_to_edges(best_adj)
        
    # Strategy 2: Multiple random restarts
    # We have ~100s left.
    # Each restart: ~5s greedy + ~5s local search = 10s.
    # We can do ~10 restarts.
    
    for _ in range(10):
        if time.time() > deadline - 15:
            break
            
        density = random.uniform(0.2, 0.6)
        adj = _random_start(density)
        
        # Greedy fill
        adj = _greedy_fill(adj, deadline - 10)
        
        # Local search
        adj = _local_search(adj, deadline - 5)
        
        count = _edge_count(adj)
        if count > best_count:
            best_adj = adj[:]
            best_count = count
            
    return _adj_to_edges(best_adj)

def _adj_to_edges(adj):
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
