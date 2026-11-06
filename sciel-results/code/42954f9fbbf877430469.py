import time
import random
import sys

def _iter_bits(mask):
    """Generator that yields set bit positions in a mask."""
    while mask:
        yield (mask & -mask).bit_length() - 1
        mask &= mask - 1

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
        return sum(a.bit_count() for a in adj) // 2

    def can_add(u, v):
        """
        Returns True if adding edge (u,v) does not create a C6.
        Optimized: iterate over the smaller neighbor set first.
        """
        Nv = adj[v]
        Nu = adj[u]
        
        # Count bits to determine which to iterate first
        if Nv.bit_count() > Nu.bit_count():
            # Swap roles: iterate over Nu (w) first, then Nv (x)
            for w in _iter_bits(Nu):
                if w == v: continue
                for x in _iter_bits(Nv):
                    if x == u or x == w: continue
                    # S_x = N(x) \ {u, v, w}
                    S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    # S_w = N(w) \ {u, v, x}
                    S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                    # Check edge between S_x and S_w
                    for y in _iter_bits(S_x):
                        if adj[y] & (S_w & ~(1 << y)):
                            return False
        else:
            # Iterate over Nv (x) first, then Nu (w)
            for x in _iter_bits(Nv):
                if x == u: continue
                for w in _iter_bits(Nu):
                    if w == v or w == x: continue
                    # S_x = N(x) \ {u, v, w}
                    S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    # S_w = N(w) \ {u, v, x}
                    S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                    # Check edge between S_x and S_w
                    for y in _iter_bits(S_x):
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
    
    # Phase 1: Hybrid Greedy - mix of random and degree-based orderings
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for trial in range(120):
        if time_left() < 8:
            break
            
        adj_try = [0] * n
        ec = 0
        
        # Mix strategies: random, degree-based, or hybrid
        strategy = trial % 3
        if strategy == 0:
            # Pure random
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
        elif strategy == 1:
            # Degree-based: prefer pairs where at least one endpoint has low degree
            # This tends to create more balanced graphs
            def degree_score(pair):
                u, v = pair
                # Prefer pairs with lower current degree sum
                return (adj_try[u].bit_count() + adj_try[v].bit_count(), random.random())
            # We can't use dynamic sorting efficiently, so use a heuristic:
            # Sort by vertex index to create structure, then add randomness
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
        else:
            # Start with a structured base (e.g., complete bipartite-ish) then random
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time_left() < 3:
                break
            
            if can_add(u, v):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]

    # Fallback: 8 disjoint K5s if greedy failed
    if best_edges == 0:
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k+a
                    v = 5*k+b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 8 * 10

    # Phase 2: Local Search with adaptive escape
    adj = best_adj[:]
    current_edges = best_edges
    
    no_improve_count = 0
    stuck_threshold = 20  # Start with lower threshold
    
    while time_left() > 2:
        # Try to add an edge
        added = False
        
        # Try random non-edges
        attempts = 30 + min(no_improve_count // 5, 20)  # Increase attempts when stuck
        for _ in range(attempts):
            if time_left() < 2:
                break
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                continue
            
            if can_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                added = True
                no_improve_count = 0
                break
        
        if not added:
            no_improve_count += 1
            
            # Escape mechanism: remove an edge when stuck
            # Lower threshold as we get more stuck
            if no_improve_count >= stuck_threshold:
                # Remove a random edge
                u = random.randint(0, n-1)
                mask = adj[u]
                if mask:
                    v = (mask & -mask).bit_length() - 1
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1
                    no_improve_count = 0
                else:
                    # Find any edge
                    for i in range(n):
                        if adj[i]:
                            u = i
                            v = (adj[i] & -adj[i]).bit_length() - 1
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            current_edges -= 1
                            break
                    else:
                        break
            
            # Gradually increase stuck threshold to allow longer exploration
            if no_improve_count == 0 and stuck_threshold < 50:
                stuck_threshold += 2

    # Update best if improved
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = adj[:]

    return get_edges()
