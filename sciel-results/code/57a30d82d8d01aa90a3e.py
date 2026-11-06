import random
import time
import numpy as np

def construct():
    N = 40
    deadline = time.time() + 108  # safety margin for 150s limit
    
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) creates a C6."""
        nu = adj[u]
        nv = adj[v]
        
        # Iterate a in N(u)
        nu_iter = nu
        while nu_iter:
            a_bit = nu_iter & (-nu_iter)
            a = a_bit.bit_length() - 1
            nu_iter ^= a_bit
            
            # Iterate b in N(v) \ {a}
            nv_iter = nv & ~(1 << a)
            while nv_iter:
                b_bit = nv_iter & (-nv_iter)
                b = b_bit.bit_length() - 1
                nv_iter ^= b_bit
                
                mask = (1 << u) | (1 << v) | (1 << a) | (1 << b)
                Sa = adj[a] & ~mask
                Sb = adj[b] & ~mask
                
                if Sa == 0 or Sb == 0:
                    continue
                
                # Check if there is an edge (x, y) with x in Sa, y in Sb, x != y
                sa_iter = Sa
                while sa_iter:
                    x_bit = sa_iter & (-sa_iter)
                    x = x_bit.bit_length() - 1
                    sa_iter ^= x_bit
                    
                    # If x is connected to any node in Sb (other than x), we have a cycle
                    if adj[x] & Sb & ~(1 << x):
                        return True
        return False

    def can_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph C6-free."""
        if adj[u] & (1 << v):
            return True  # already present
        return not would_create_c6(adj, u, v)

    def local_search(adj, edge_set, time_limit):
        """1-edge-swap local search: try removing one edge and adding another."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try random 1-swaps
            for _ in range(500):
                if time.time() >= time_limit:
                    break
                # Pick a random existing edge to remove
                e_remove = random.choice(list(edge_set))
                u1, v1 = e_remove
                # Pick a random non-edge to add
                u2, v2 = random.choice(all_edges)
                if (u2, v2) in edge_set or (v2, u2) in edge_set:
                    continue
                # Check if we can remove e_remove and add (u2,v2)
                # First check: can we add (u2,v2) without e_remove? No, we need to remove first.
                # Simulate removal
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                edge_set.discard(e_remove)
                edge_set.discard((v1, u1))
                
                # Now try to add (u2, v2)
                if can_add(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    edge_set.add((u2, v2))
                    improved = True
                else:
                    # Revert
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    edge_set.add(e_remove)
        
        return len(edge_set)

    # Randomized Greedy + Local Search
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        adj = [0] * N
        edge_set = set()
        
        for u, v in all_edges:
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        current_count = len(edge_set)
        
        # Apply local search if we have time
        if time.time() < deadline - 2:
            # Use remaining time minus a buffer for local search
            ls_deadline = min(deadline - 1, time.time() + 5)
            current_count = local_search(adj, edge_set, ls_deadline)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edge_set)
            
            if best_count > 105:
                pass

    return [[u, v] for u, v in best_edges]
