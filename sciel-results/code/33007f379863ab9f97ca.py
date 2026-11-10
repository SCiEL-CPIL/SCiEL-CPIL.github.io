import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 20
    
    def popcount(x):
        return x.bit_count()
    
    def is_valid_add(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
            
        # Check 1: u and v cannot have >= 3 common neighbors
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False
            
        # Check 2: For any w in N(u), v and w cannot have >= 2 common neighbors
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            mask_u ^= lsb
            if w == v: continue
            common_wv = adj[w] & adj[v]
            if popcount(common_wv) >= 2:
                return False
                
        # Check 3: For any x in N(v), u and x cannot have >= 2 common neighbors
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            mask_v ^= lsb
            if x == u: continue
            common_ux = adj[u] & adj[x]
            if popcount(common_ux) >= 2:
                return False
                
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def try_add_all_missing(adj, all_edges):
        """Try to add any missing edge that's valid. Returns number added."""
        added = 0
        changed = True
        while changed:
            changed = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        added += 1
                        changed = True
                        if time.time() > deadline:
                            return added
        return added

    def local_search(adj, all_edges, max_iter=500):
        """Local search with edge swaps to escape local optima."""
        current_edges = count_edges(adj)
        best_adj = adj.copy()
        best_edges = current_edges
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to add any missing edge
            added_any = try_add_all_missing(adj, all_edges)
            if added_any > 0:
                current_edges += added_any
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj.copy()
                continue
            
            # Stuck: try edge swap - remove one edge, try to add up to 2 new edges
            # Pick a random existing edge to remove
            u = random.randint(0, n - 1)
            if not adj[u]:
                continue
            mask = adj[u]
            # Get list of neighbors
            neighbors = []
            tmp = mask
            while tmp:
                lsb = tmp & -tmp
                w = lsb.bit_length() - 1
                neighbors.append(w)
                tmp ^= lsb
            
            if not neighbors:
                continue
            v = random.choice(neighbors)
            
            # Save state
            old_adj = adj.copy()
            remove_edge(adj, u, v)
            
            # Try to add edges
            added = try_add_all_missing(adj, all_edges)
            new_edges = current_edges - 1 + added
            
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = adj.copy()
                current_edges = new_edges
            else:
                # Revert
                adj = old_adj
        
        return best_adj, best_edges

    # Main search
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_adj = [0] * n
    best_edges = 0
    
    # Strategy 1: Greedy with random restarts
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        adj = [0] * n
        
        # Randomize edge order
        random.shuffle(all_edges)
        
        # Greedy add
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
        
        current_edges = count_edges(adj)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj.copy()
        
        # Local search to improve
        if current_edges >= best_edges - 2:  # Only if close to best
            improved_adj, improved_edges = local_search(adj, all_edges, max_iter=100)
            if improved_edges > best_edges:
                best_edges = improved_edges
                best_adj = improved_adj
    
    # Strategy 2: Try to build from structured initializations
    # Complete bipartite K_{10,10} has 100 edges but contains K_{2,3}
    # Let's try near-regular graphs
    
    # Try a random regular-like construction
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        
        # Randomly assign degrees ~10
        for i in range(n):
            # Try to add ~10 random neighbors
            candidates = list(range(n))
            candidates.remove(i)
            random.shuffle(candidates)
            
            for v in candidates:
                if (adj[i] >> v) & 1:
                    continue
                if is_valid_add(adj, i, v):
                    add_edge(adj, i, v)
                if adj[i].bit_count() >= 10:
                    break
        
        current_edges = count_edges(adj)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj.copy()
        
        # Local search
        if current_edges >= best_edges - 3:
            improved_adj, improved_edges = local_search(adj, all_edges, max_iter=200)
            if improved_edges > best_edges:
                best_edges = improved_edges
                best_adj = improved_adj
    
    # Final saturation
    if time.time() < deadline - 5:
        try_add_all_missing(best_adj, all_edges)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges
    
    # Return edges
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            mask ^= lsb
            if i < j:
                edges.append([i, j])
                
    return edges
