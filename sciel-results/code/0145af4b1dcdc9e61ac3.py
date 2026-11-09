import time
import random

def construct():
    n = 20
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
            
        # Check neighbors of u
        # For each w in N(u), we need common(w, v) < 2
        # common(w, v) is popcount(adj[w] & adj[v])
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if popcount(adj[w] & adj[v]) >= 2:
                return False
            mask_u ^= lsb
            
        # Check neighbors of v
        # For each x in N(v), we need common(u, x) < 2
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            if popcount(adj[u] & adj[x]) >= 2:
                return False
            mask_v ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges():
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
    
    def get_existing_edges():
        existing = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    existing.append((i, j))
                mask ^= lsb
        return existing
    
    def try_1_for_2_swap():
        """Try to remove one edge and add two others (net +1)."""
        existing = get_existing_edges()
        if not existing:
            return False
        
        # Try each existing edge as the one to remove
        random.shuffle(existing)
        for rem_u, rem_v in existing:
            # Temporarily remove
            remove_edge(rem_u, rem_v)
            
            # Try to add two new edges
            added = []
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if (u, v) == (rem_u, rem_v) or (u, v) == (rem_v, rem_u):
                    continue
                if not ((adj[u] >> v) & 1):
                    if can_add(u, v):
                        add_edge(u, v)
                        added.append((u, v))
                        if len(added) == 2:
                            break
            
            if len(added) == 2:
                # Net gain of +1
                return True
            else:
                # Undo what we added
                for u, v in added:
                    remove_edge(u, v)
                # Re-add the removed edge
                add_edge(rem_u, rem_v)
        
        return False
    
    def final_polish():
        """Exhaustively check all 1-for-2 swaps and single additions."""
        improved = True
        while improved:
            improved = False
            
            # Try all single additions
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            for u, v in pairs:
                if not ((adj[u] >> v) & 1):
                    if can_add(u, v):
                        add_edge(u, v)
                        improved = True
            
            # Try all 1-for-2 swaps
            existing = get_existing_edges()
            for rem_u, rem_v in existing:
                remove_edge(rem_u, rem_v)
                
                added = []
                pairs2 = [(i, j) for i in range(n) for j in range(i+1, n)]
                for u, v in pairs2:
                    if (u, v) == (rem_u, rem_v):
                        continue
                    if not ((adj[u] >> v) & 1):
                        if can_add(u, v):
                            add_edge(u, v)
                            added.append((u, v))
                            if len(added) == 2:
                                break
                
                if len(added) == 2:
                    improved = True
                else:
                    for u, v in added:
                        remove_edge(u, v)
                    add_edge(rem_u, rem_v)
        
        return count_edges()

    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer
    
    best_adj = [0] * n
    best_count = 0
    
    # Multiple restarts to explore different local optima
    max_restarts = 5
    time_per_restart = (deadline - start_time) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30:  # Save 30s for final polish
            break
        
        # Reset
        adj = [0] * n
        
        # Random initialization: greedily add edges in random order
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if can_add(u, v):
                add_edge(u, v)
        
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Local search with 1-for-2 swaps
        restart_deadline = start_time + (restart + 1) * time_per_restart
        while time.time() < restart_deadline:
            # Try to add any missing edge
            added = False
            random.shuffle(pairs)
            for u, v in pairs:
                if not ((adj[u] >> v) & 1):
                    if can_add(u, v):
                        add_edge(u, v)
                        added = True
            
            current_count = count_edges()
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
            
            if not added:
                # Stuck, try 1-for-2 swap
                if not try_1_for_2_swap():
                    # Swap didn't help, remove random edge
                    existing = get_existing_edges()
                    if existing:
                        u, v = random.choice(existing)
                        remove_edge(u, v)
                    else:
                        break
        
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
    
    # Final polish: run on the best graph found
    adj = best_adj[:]
    final_count = final_polish()
    if final_count > best_count:
        best_count = final_count
        best_adj = adj[:]
    
    # Return the best graph found
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges
