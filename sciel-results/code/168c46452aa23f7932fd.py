import time
import random
import numpy as np
from collections import defaultdict

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115.0
    
    # We'll use bitmasks for adjacency
    # adj[i] is a bitmask where bit j is set if (i,j) is an edge
    
    def compute_book(adj_mask):
        """Compute max book size for a graph represented by bitmask adjacency."""
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    common = bin(adj_mask[i] & adj_mask[j]).count('1')
                    if common > max_book:
                        max_book = common
        return max_book
    
    def count_edges(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    # Initial construction: balanced multipartite + extra edges
    # Start with K20,20 (400 edges)
    adj = [0] * n
    for a in range(20):
        for b in range(20, 40):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    # Add one extra edge within part 0
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    
    current_max = compute_book(adj)
    best_max = current_max
    best_adj = adj[:]
    
    random.seed(42)
    
    # Hill climbing with edge swaps
    while time.time() < deadline and current_max > 6:
        # Get list of edges and non-edges
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        improved = False
        attempts = 0
        max_attempts = 500
        
        while attempts < max_attempts and not improved and time.time() < deadline:
            attempts += 1
            # Pick random edge to remove
            rem_u, rem_v = random.choice(edges)
            # Pick random non-edge to add
            new_u, new_v = random.choice(non_edges)
            
            if new_u == rem_u and new_v == rem_v:
                continue
            
            # Apply swap
            adj[rem_u] &= ~(1 << rem_v)
            adj[rem_v] &= ~(1 << rem_u)
            adj[new_u] |= (1 << new_v)
            adj[new_v] |= (1 << new_u)
            
            new_max = compute_book(adj)
            
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_adj = adj[:]
                improved = True
                # Rebuild edge/non-edge lists
                edges = []
                non_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges.append((i, j))
                        else:
                            non_edges.append((i, j))
            else:
                # Revert
                adj[new_u] &= ~(1 << new_v)
                adj[new_v] &= ~(1 << new_u)
                adj[rem_u] |= (1 << rem_v)
                adj[rem_v] |= (1 << rem_u)
        
        if not improved:
            break
    
    # Return the best configuration found
    result_adj = best_adj if best_max <= current_max else adj
    
    # Build edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (result_adj[i] >> j) & 1:
                result.append([i, j])
    
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    return result
