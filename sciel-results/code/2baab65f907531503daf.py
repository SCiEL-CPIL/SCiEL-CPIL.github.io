import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, u, v):
        # adj is a list of integers (bitmasks)
        # 1. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 2. Check constraints on neighbors of u
        # For w in N(u), u becomes a common neighbor of v and w.
        # So we need current common(v, w) <= 1.
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            # Calculate common neighbors of v and w
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 3. Check constraints on neighbors of v
        # For w in N(v), v becomes a common neighbor of u and w.
        # So we need current common(u, w) <= 1.
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_fill(order, adj=None):
        if adj is None:
            adj = [0] * n
        count = 0
        # Calculate current count if adj is provided
        if adj is not None and any(adj):
            for i in range(n):
                count += adj[i].bit_count()
            count //= 2
            
        for u, v in order:
            if not (adj[u] >> v) & 1: # Check if edge already exists
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
        return adj, count

    def local_search(adj, count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Add any missing edge
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        missing.append((i, j))
                        
            if missing:
                random.shuffle(missing)
                for u, v in missing:
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
                        improved = True
                        break # Restart loop to update missing list? 
                        # Breaking and restarting the while loop is safer for correctness of 'missing' list
                        # But it's expensive to rebuild 'missing'. 
                        # Let's just continue and mark as improved, but we need to be careful not to add invalid ones later in the same pass.
                        # Actually, adding one edge might invalidate others. 
                        # So restarting the pass is correct.
                        if time.time() > deadline:
                            return adj, count
                        break # Break the for loop, continue while loop
            
            if improved:
                continue
            
            # Phase 2: If stuck, try removing an edge to unlock others
            # Find existing edges
            existing = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        existing.append((i, j))
                    mask ^= lsb
                    
            if not existing:
                break
                
            random.shuffle(existing)
            for u, v in existing:
                # Try removing (u, v)
                remove_edge(adj, u, v)
                count -= 1
                
                # Try to add back any missing edges
                new_count = count
                added_any = False
                # We can reuse a simple loop here
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] >> j) & 1:
                            if is_valid_add(adj, i, j):
                                add_edge(adj, i, j)
                                new_count += 1
                                added_any = True
                                if time.time() > deadline:
                                    return adj, new_count
                
                if new_count > count: # count here is pre-removal count? No, count was decremented.
                    # We need to compare new_count with the count BEFORE removal
                    # Let's store count_before
                    pass 
                # This logic is getting messy with counts.
                # Let's simplify: 
                # Store adj_copy? Too expensive.
                # Just track count carefully.
                
                # Let's restart the local_search loop if we found an improvement?
                # Or just accept it and continue.
                
                # If new_count > count_before_removal (which was count+1), then we improved.
                # count_before_removal = count + 1
                if new_count > count + 1:
                    # Keep the change
                    count = new_count
                    improved = True
                    break # Restart while loop
                else:
                    # Revert
                    add_edge(adj, u, v)
                    count += 1
                    # Try next edge to remove
                    if time.time() > deadline:
                        return adj, count

        return adj, count

    # Run multiple random restarts
    restarts = 0
    # Try a few deterministic orders first
    deterministic_orders = [
        all_edges,
        sorted(all_edges, key=lambda x: (x[0] + x[1], x[0])),
        sorted(all_edges, reverse=True)
    ]
    
    for order in deterministic_orders:
        if time.time() > deadline - 5:
            break
        adj, count = greedy_fill(order)
        adj, count = local_search(adj, count)
        
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        best_edges.append([i, j])
                    mask ^= lsb

    while time.time() < deadline - 5:
        random.shuffle(all_edges)
        adj, count = greedy_fill(all_edges)
        adj, count = local_search(adj, count)
        
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        best_edges.append([i, j])
                    mask ^= lsb
                    
        restarts += 1
        
        if time.time() > deadline - 10:
            break

    return best_edges
