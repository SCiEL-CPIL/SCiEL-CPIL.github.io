"""Optimized K2,3-free graph construction on 20 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute common neighbors bitmasks for speed? 
    # Actually, we can just use bit operations on adjacency lists.
    
    def count_common_neighbors(adj, u, v):
        return bin(adj[u] & adj[v]).count('1')
        
    def can_add_edge(adj, u, v):
        if u == v or ((adj[u] >> v) & 1):
            return False
        # Check if adding (u,v) creates a K2,3
        # A K2,3 is formed if there are 3 vertices w such that w is neighbor to both u and v.
        # So we just need count_common_neighbors(adj, u, v) < 3.
        if count_common_neighbors(adj, u, v) >= 3:
            return False
        return True
        
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def get_edge_count(adj):
        # Sum of degrees / 2
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2
        
    def is_valid_graph(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if ((adj[u] >> v) & 1):
                    if count_common_neighbors(adj, u, v) >= 3:
                        return False
        return True

    best_adj = [0] * n
    best_count = 0
    rng = random.Random(42)
    
    # Strategy 1: Randomized Greedy Construction with Local Search
    # We run multiple iterations to find a good starting point and then improve.
    
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
            
        adj = [0] * n
        edges_list = all_edges[:]
        rng.shuffle(edges_list)
        
        # Greedy Phase
        for u, v in edges_list:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
                
        # Local Search Phase: Try to add edges
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
        
        # Local Search Phase: Try to swap edges to increase count
        # A simple swap: remove e1, add e2, add e3? 
        # Or just try to find a move that increases count.
        # Since adding is already maximized, we look for swaps.
        # Swap: Remove (u,v), Add (u,w), Add (v,x)? No, that changes structure too much.
        # Let's try a 1-for-1 swap first? No, that keeps count same.
        # We need a move that increases count.
        # Since we are in a local max for additions, we must remove at least one edge.
        # To increase count, we need to add more edges than we remove.
        # E.g., Remove 1, Add 2.
        
        # Heuristic: Try removing each edge and see if we can add 2 new ones.
        # This is O(E * E^2) which is too slow for 190 edges if done naively.
        # But n=20 is small.
        
        # Let's try a simpler improvement:
        # If we can't add any edge, try to remove one edge (u,v) and see if we can add two new edges.
        # We'll limit this search to avoid timeout.
        
        current_count = get_edge_count(adj)
        
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
            
        # If we have time, do a more expensive local search
        if time.time() < deadline - 10:
            # Try to find a beneficial swap
            # We iterate over edges to remove
            for u, v in all_edges:
                if ((adj[u] >> v) & 1):
                    # Remove (u,v)
                    remove_edge(adj, u, v)
                    
                    # Try to add edges
                    added = 0
                    added_edges = []
                    for u2, v2 in all_edges:
                        if u2 == u and v2 == v:
                            continue
                        if not ((adj[u2] >> v2) & 1) and can_add_edge(adj, u2, v2):
                            add_edge(adj, u2, v2)
                            added_edges.append((u2, v2))
                            added += 1
                            if added == 2: # Enough to gain 1 edge
                                break
                            
                    if added > 1:
                        # We gained 'added - 1' edges
                        new_count = current_count - 1 + added
                        if new_count > best_count:
                            best_adj = adj[:]
                            best_count = new_count
                    else:
                        # Rollback
                        for u2, v2 in added_edges:
                            remove_edge(adj, u2, v2)
                        add_edge(adj, u, v)
                    
                    if time.time() > deadline - 5:
                        break
                    
    # Final verification and extraction
    if not is_valid_graph(best_adj):
        # Fallback to a simple valid graph if best_adj is corrupted (shouldn't happen)
        best_adj = [0] * n
        for u, v in all_edges:
            if can_add_edge(best_adj, u, v):
                add_edge(best_adj, u, v)
                
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result_edges.append([u, v])
                
    return result_edges

# EVOLVE-BLOCK-END
