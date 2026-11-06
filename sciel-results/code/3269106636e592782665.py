import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 115  # Safety margin: 5s buffer before 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bitset adjacency
    # adj[i] is an integer where bit j is set if (i,j) is an edge
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count():
        return sum(bin(a).count("1") for a in adj) // 2

    def can_add(u, v):
        """
        Returns True if adding edge (u,v) does not create a C6.
        C6: u-v-x-y-z-w-u where all vertices distinct.
        x in N(v)\{u}, w in N(u)\{v}, y in N(x)\{u,v,w}, z in N(w)\{u,v,x}, y~z, y!=z.
        """
        Nv = adj[v] & ~(1 << u)
        Nu = adj[u] & ~(1 << v)
        
        # Iterate over the smaller neighbor set first
        if bin(Nv).count('1') <= bin(Nu).count('1'):
            outer_mask = Nv
            inner_mask_template = Nu
            outer_is_v = True
        else:
            outer_mask = Nu
            inner_mask_template = Nv
            outer_is_v = False
        
        while outer_mask:
            outer = (outer_mask & -outer_mask).bit_length() - 1
            outer_mask &= outer_mask - 1
            
            inner_mask = inner_mask_template & ~(1 << outer)
            while inner_mask:
                inner = (inner_mask & -inner_mask).bit_length() - 1
                inner_mask &= inner_mask - 1
                
                if outer_is_v:
                    x, w = outer, inner
                else:
                    x, w = inner, outer
                
                # S_x = N(x) \ {u, v, w}
                S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                # S_w = N(w) \ {u, v, x}
                S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                
                # Check if there is an edge between S_x and S_w with y != z
                y_mask = S_x
                while y_mask:
                    y = (y_mask & -y_mask).bit_length() - 1
                    y_mask &= y_mask - 1
                    
                    if adj[y] & (S_w & ~(1 << y)):
                        return False
        return True

    def get_edges():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    random.seed(42)
    
    best_edges = 0
    best_adj = [0] * n
    
    # Phase 1: Random Greedy with degree-based ordering
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for trial in range(80):
        if time_left() < 10:
            break
            
        adj_try = [0] * n
        ec = 0
        degrees = [0] * n
        
        # Random shuffle
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time_left() < 5:
                break
            
            if can_add(u, v):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                ec += 1
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]

    # If we didn't find anything good, use a simple structure (e.g., union of cliques)
    # But greedy usually finds something.
    if best_edges == 0:
        # Fallback: 8 disjoint K5s
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k+a
                    v = 5*k+b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 8 * 10  # K5 has 10 edges

    # Phase 2: Local Search
    adj = best_adj[:]
    current_edges = best_edges
    
    # Precompute list of non-edges for faster sampling
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
    
    no_improve_count = 0
    last_improve_time = time.time()
    
    while time_left() > 2:
        # Try to add an edge
        added = False
        
        # Try more random non-edges per iteration
        for _ in range(40):
            if time_left() < 2:
                break
            if not non_edges:
                break
            u, v = random.choice(non_edges)
            # Verify it's still a non-edge (should be, but safety check)
            if adj[u] & (1 << v):
                continue
            
            if can_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                added = True
                no_improve_count = 0
                last_improve_time = time.time()
                # Rebuild non_edges list
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                break
        
        if not added:
            # Try to remove an edge to escape local maximum
            # Only remove if we've been stuck for a while
            if no_improve_count > 30:
                # Remove a random edge
                if current_edges > 0:
                    # Pick random u, then random v in adj[u]
                    u = random.randint(0, n-1)
                    mask = adj[u]
                    if mask:
                        v = (mask & -mask).bit_length() - 1
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        current_edges -= 1
                        no_improve_count = 0
                        # Rebuild non_edges list
                        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                    else:
                        # Find any edge
                        found = False
                        for i in range(n):
                            if adj[i]:
                                u = i
                                v = (adj[i] & -adj[i]).bit_length() - 1
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                current_edges -= 1
                                found = True
                                no_improve_count = 0
                                # Rebuild non_edges list
                                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                                break
                        if not found:
                            break
            
            no_improve_count += 1

    # Update best if improved
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = adj[:]

    return get_edges()
