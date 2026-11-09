import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Fallback: Andrasfai graph And(9) on 26 vertices
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # We try n=40, then n=39, etc.
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Try to find a valid graph on n vertices
        found = local_search_for_valid_graph(n, deadline)
        if found:
            return found
            
    return best_result

def local_search_for_valid_graph(n, deadline):
    # Initialize random maximal triangle-free graph
    adj = [0] * n
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(pairs)
    
    for u, v in pairs:
        if (adj[u] & adj[v]) == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    # Precompute non-adjacency masks for faster access? 
    # adj[i] is mask of neighbors.
    
    # Estimate initial alpha
    alpha_curr = estimate_alpha(adj, n)
    
    if alpha_curr < 10:
        if exact_has_independent_set_10(adj, n):
            return None # Wait, exact says it HAS independent set? 
            # No, exact_has_independent_set_10 returns True if alpha >= 10.
            # So if alpha_curr < 10 (greedy), we check exact.
            # If exact returns False, then alpha < 10 is confirmed.
            # If exact returns True, then greedy was wrong.
            pass
        else:
            # Greedy said < 10, Exact says < 10. Valid!
            edges = extract_edges(adj, n)
            return {'n': n, 'edges': edges}
            
    # Local Search
    # We want to minimize alpha.
    # Move: remove random edge (u,v), add random non-edge (x,y) if triangle-free.
    
    # To pick random non-edge efficiently, we can precompute list of non-edges?
    # Or just pick random x,y and check if not edge.
    
    while time.time() < deadline - 2:
        # Pick random edge to remove
        # Count edges
        edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
        if edge_count == 0:
            break
            
        # Pick random edge
        # This is inefficient if we do it naively.
        # Better: maintain list of edges? Or just pick random u, then random neighbor v.
        u = random.randint(0, n - 1)
        if adj[u] == 0:
            continue
        v = (adj[u].bit_length() - 1 - random.randint(0, adj[u].bit_count() - 1))
        # The above line is wrong for bit selection.
        # Let's do this:
        mask = adj[u]
        # Pick random bit set in mask
        # random.shuffle is not good for bits.
        # Use random.getrandbits?
        # Simple way:
        bits = []
        m = mask
        while m:
            lsb = m & -m
            bits.append(lsb.bit_length() - 1)
            m ^= lsb
        if not bits:
            continue
        v = random.choice(bits)
        
        # Remove edge (u,v)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Pick random non-edge (x,y)
        # Try a few times to find a valid non-edge that doesn't create triangle
        success = False
        for _ in range(50): # Max attempts to find a valid swap
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            if (adj[x] & (1 << y)) != 0:
                continue # Already an edge
            
            # Check if adding (x,y) creates triangle
            # Triangle if neighbors[x] & neighbors[y] != 0
            if (adj[x] & adj[y]) == 0:
                # Safe to add
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
                
                # Calculate new alpha
                alpha_new = estimate_alpha(adj, n)
                
                if alpha_new < 10:
                    # Check exact
                    if not exact_has_independent_set_10(adj, n):
                        edges = extract_edges(adj, n)
                        return {'n': n, 'edges': edges}
                    # If exact says alpha >= 10, we accepted the move because it was "valid" triangle-free
                    # but we might want to revert if it increased alpha?
                    # For now, keep it if it's triangle free.
                
                # Heuristic: Accept if alpha_new <= alpha_curr (non-increasing)
                # Or with some probability if alpha_new > alpha_curr (annealing)
                # For simplicity, accept if alpha_new <= alpha_curr
                if alpha_new <= alpha_curr:
                    alpha_curr = alpha_new
                    success = True
                    break
                else:
                    # Revert
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
        
        if not success:
            # Revert the removal of (u,v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return None

def estimate_alpha(adj, n):
    """
    Greedy estimate of independence number.
    """
    all_mask = (1 << n) - 1
    non_adj = [0] * n
    for i in range(n):
        non_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
        
    mask = all_mask
    size = 0
    while mask:
        if size >= 10:
            return size # Early exit
        
        # Pick vertex with min degree in induced subgraph
        min_deg = n + 1
        best_v = -1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            break
            
        size += 1
        mask &= non_adj[best_v]
        
    return size

def exact_has_independent_set_10(adj, n):
    """
    Returns True if there exists an independent set of size 10.
    """
    all_mask = (1 << n) - 1
    non_adj = [0] * n
    for i in range(n):
        non_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
        
    def dfs(mask, size):
        if size == 10:
            return True
        if mask == 0:
            return False
        if mask.bit_count() + size < 10:
            return False
        
        # Heuristic: pick vertex with max degree in induced subgraph
        max_deg = -1
        best_v = -1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            return False
        
        v_bit = 1 << best_v
        
        # Include best_v
        if dfs(non_adj[best_v] & mask, size + 1):
            return True
        
        # Exclude best_v
        if dfs(mask & ~v_bit, size):
            return True
        
        return False
    
    return dfs(all_mask, 0)

def extract_edges(adj, n):
    edges = []
    for i in range(n):
        mask = adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
    return edges
