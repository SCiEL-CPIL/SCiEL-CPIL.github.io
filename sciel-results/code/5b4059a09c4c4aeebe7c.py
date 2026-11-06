import random
import time

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 145
    
    adj = [0] * n
    
    def add_edge(u, v):
        if u > v: u, v = v, u
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        if u > v: u, v = v, u
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def has_edge(u, v):
        if u > v: u, v = v, u
        return (adj[u] >> v) & 1
        
    def get_book_size(u, v):
        if u > v: u, v = v, u
        if not has_edge(u, v): return -1
        return bin(adj[u] & adj[v]).count('1')
        
    def get_max_book():
        max_bk = 0
        max_edge = None
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    bk = bin(adj[u] & adj[v]).count('1')
                    if bk > max_bk:
                        max_bk = bk
                        max_edge = (u, v)
                mask &= mask - 1
        return max_bk, max_edge

    # Initialize
    for a in range(20):
        for b in range(20, 40):
            add_edge(a, b)
    add_edge(0, 1)
    
    current_max, current_edge = get_max_book()
    best_max = current_max
    best_adj = adj[:]
    
    no_improve_count = 0
    
    while time.time() < deadline:
        current_max, current_edge = get_max_book()
        if current_max < best_max:
            best_max = current_max
            best_adj = adj[:]
            no_improve_count = 0
            if best_max <= 7:
                break
        else:
            no_improve_count += 1
            
        # If stuck for too long, restart with a random perturbation
        if no_improve_count > 500:
            # Random restart: keep 80% of edges, randomize 20%
            # Or just random swap
            for _ in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if has_edge(u, v):
                    remove_edge(u, v)
                    # Add random non-edge
                    while True:
                        a = random.randint(0, n-1)
                        b = random.randint(0, n-1)
                        if a == b: continue
                        if a > b: a, b = b, a
                        if not has_edge(a, b):
                            add_edge(a, b)
                            break
            no_improve_count = 0
            continue

        u, v = current_edge
        
        # Find common neighbors
        common_mask = adj[u] & adj[v]
        if common_mask == 0:
            break
            
        # Get list of common neighbors
        common_neighbors = []
        mask = common_mask
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            common_neighbors.append(w)
            mask &= mask - 1
            
        if not common_neighbors:
            break
            
        # Pick random common neighbor
        w = random.choice(common_neighbors)
        
        # Remove (u, w)
        old_adj = adj[:]
        remove_edge(u, w)
        
        # Find best non-edge to add via sampling
        best_new_edge = None
        best_new_book = current_max + 1
        
        # Sample 50 non-edges
        for _ in range(50):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b: continue
            if a > b: a, b = b, a
            if has_edge(a, b): continue
            
            bk = bin(adj[a] & adj[b]).count('1')
            if bk < best_new_book:
                best_new_book = bk
                best_new_edge = (a, b)
                
        if best_new_edge is None:
            adj = old_adj
            continue
            
        # Accept if it doesn't increase max book
        # Note: current_max is the book of (u, v).
        # After removing (u, w), book(u, v) is current_max - 1.
        # If best_new_book <= current_max - 1, new max is <= current_max - 1.
        # If best_new_book == current_max, new max is current_max.
        # If best_new_book > current_max, new max is best_new_book.
        
        if best_new_book <= current_max:
            add_edge(*best_new_edge)
        else:
            adj = old_adj

    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])
                
    return result
