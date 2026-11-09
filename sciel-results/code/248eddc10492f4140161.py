import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    n = 40
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    for restart in range(10000):
        if time.time() > deadline:
            break
        
        # 1. Greedy Construction
        edges_order = all_edges[:]
        random.shuffle(edges_order)
        
        adj = [0] * n
        current_edges = []
        
        for u, v in edges_order:
            if time.time() > deadline:
                break
            if not _creates_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
        
        # 2. Local Search: Hill Climbing with Swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            if not current_edges:
                break
            
            # Pick a random edge to remove
            idx = random.randint(0, len(current_edges) - 1)
            e_rem = current_edges[idx]
            u, v = e_rem
            
            old_size = len(current_edges)
            
            # Remove it
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges.pop(idx)
            
            edge_set = set(current_edges)
            
            candidates = [e for e in all_edges if e not in edge_set]
            random.shuffle(candidates)
            
            added_edges = []
            for x, y in candidates:
                if time.time() > deadline:
                    break
                if (x, y) in edge_set:
                    continue
                if not _creates_c6(adj, x, y):
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    current_edges.append((x, y))
                    edge_set.add((x, y))
                    added_edges.append((x, y))
            
            new_size = len(current_edges)
            
            if new_size > old_size:
                improved = True
            else:
                # Revert: Remove added edges, Re-add e_rem
                for ax, ay in added_edges:
                    adj[ax] &= ~(1 << ay)
                    adj[ay] &= ~(1 << ax)
                    # Remove from current_edges
                    # This is O(N) for each removal. 
                    # To be efficient, we could just rebuild current_edges from adj?
                    # Or use a set and then convert to list?
                    # Given the small size, O(N^2) is fine.
                    try:
                        current_edges.remove((ax, ay))
                    except ValueError:
                        pass # Should not happen
                    
                # Re-add e_rem
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append(e_rem)

        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]

    return [list(e) for e in best_edges]

def _creates_c6(adj, u, v):
    """
    Check if adding edge (u, v) creates a C6.
    adj: list of integers, bitmask of neighbors.
    """
    n1s = adj[u] & ~(1 << v)
    
    while n1s:
        lsb = n1s & -n1s
        n1 = lsb.bit_length() - 1
        n1s ^= lsb
        
        n2s = adj[n1] & ~(1 << u) & ~(1 << v) & ~(1 << n1)
        
        while n2s:
            lsb = n2s & -n2s
            n2 = lsb.bit_length() - 1
            n2s ^= lsb
            
            n3s = adj[n2] & ~(1 << u) & ~(1 << v) & ~(1 << n1) & ~(1 << n2)
            
            while n3s:
                lsb = n3s & -n3s
                n3 = lsb.bit_length() - 1
                n3s ^= lsb
                
                n4s = adj[n3] & ~(1 << u) & ~(1 << v) & ~(1 << n1) & ~(1 << n2) & ~(1 << n3)
                
                while n4s:
                    lsb = n4s & -n4s
                    n4 = lsb.bit_length() - 1
                    n4s ^= lsb
                    
                    if adj[n4] & (1 << v):
                        return True
                        
    return False
