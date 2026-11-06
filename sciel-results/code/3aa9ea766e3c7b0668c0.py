import time
import random
import math

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115  # Safety margin for 120s limit

    # Initialize with K_20,20 + 1 intra-partition edge
    # Vertices 0-19 in set A, 20-39 in set B
    adj = [0] * n
    edges = set()
    
    # Add all edges between A and B
    for u in range(20):
        for v in range(20, 40):
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Add 1 edge within A to reach 401 edges
    # (0, 1) is in A
    edges.add((0, 1))
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    
    def calc_books():
        max_b = 0
        sum_b = 0
        books = {}
        # Local bindings for speed
        adj_local = adj
        edges_local = edges
        for u, v in edges_local:
            b = (adj_local[u] & adj_local[v]).bit_count()
            books[(u, v)] = b
            if b > max_b:
                max_b = b
            sum_b += b
        return max_b, sum_b, books

    max_b, sum_b, books = calc_books()
    best_edges = edges.copy()
    best_max = max_b
    best_sum = sum_b
    
    # Local Search
    # Move: Remove one edge, Add one edge.
    # We use a batch approach: try multiple random moves, pick the best one.
    
    batch_size = 20
    
    while time.time() < deadline:
        best_candidate_max = max_b
        best_candidate_sum = sum_b
        best_candidate_edges = None
        best_candidate_books = None
        
        for _ in range(batch_size):
            # Pick random edge to remove
            e_rem = random.choice(list(edges))
            u1, v1 = e_rem
            
            # Temp remove
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            edges.discard(e_rem)
            
            # Pick random non-edge to add
            while True:
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2:
                    continue
                key = (min(u2, v2), max(u2, v2))
                if key not in edges:
                    break
            
            # Temp add
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edges.add(key)
            
            # Eval
            new_max, new_sum, new_books = calc_books()
            
            # Check if better
            if new_max < best_candidate_max or (new_max == best_candidate_max and new_sum < best_candidate_sum):
                best_candidate_max = new_max
                best_candidate_sum = new_sum
                best_candidate_edges = edges.copy()
                best_candidate_books = new_books
            
            # Revert temp changes for next candidate in batch
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)
            edges.discard(key)
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            edges.add(e_rem)
            
        # End batch
        if best_candidate_edges is not None:
            edges = best_candidate_edges
            max_b = best_candidate_max
            sum_b = best_candidate_sum
            books = best_candidate_books
            
            if max_b < best_max or (max_b == best_max and sum_b < best_sum):
                best_max = max_b
                best_sum = sum_b
                best_edges = edges.copy()
                
            # If we hit the theoretical lower bound for max book (7), we stop
            if best_max == 7:
                break
        else:
            # Stagnation: try a random restart?
            # Or just continue.
            pass

    if best_edges is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    return [[u, v] for (u, v) in sorted(best_edges)]
