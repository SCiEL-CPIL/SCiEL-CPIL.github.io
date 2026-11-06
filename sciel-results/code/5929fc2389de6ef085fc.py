import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, u, v):
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

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    missing.append((i, j))
        return missing

    def local_search(adj, count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges(adj)
            if not missing:
                break
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
                    if time.time() > deadline:
                        return count
        return count

    def two_opt_swap(adj, count):
        # Remove a random edge, try to add two missing edges
        # If successful, count increases by 1
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        
        if not present:
            return count, False
            
        u, v = random.choice(present)
        remove_edge(adj, u, v)
        count -= 1
        
        # Try to add edges greedily
        missing = get_missing_edges(adj)
        random.shuffle(missing)
        added = 0
        for a, b in missing:
            if is_valid_add(adj, a, b):
                add_edge(adj, a, b)
                count += 1
                added += 1
                if added >= 2:
                    break
        if added < 2:
            # Revert if we didn't gain
            # Note: We added 'added' edges. We need to remove them.
            # This is tricky because we don't track which ones were added in this specific swap easily without a list.
            # Let's simplify: just do a full local search after removal.
            pass
        return count, True

    def simulated_annealing(adj, count):
        if time.time() > deadline:
            return count
        
        T = 2.0
        cooling_rate = 0.99
        min_T = 0.1
        
        while T > min_T and time.time() < deadline - 2:
            # Pick a random move
            if random.random() < 0.5:
                # Try to add a missing edge
                missing = get_missing_edges(adj)
                if missing:
                    u, v = random.choice(missing)
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
            else:
                # Try to remove a present edge and refill
                present = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            present.append((i, j))
                        mask ^= lsb
                
                if present:
                    u, v = random.choice(present)
                    remove_edge(adj, u, v)
                    count -= 1
                    # Refill
                    count = local_search(adj, count)
            
            T *= cooling_rate
        
        return count

    # Run multiple random restarts
    restarts = 0
    while time.time() < deadline - 5:
        random.shuffle(all_edges)
        adj = [0] * n
        count = 0
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        
        # Local search
        count = local_search(adj, count)
        
        # Simulated annealing
        count = simulated_annealing(adj, count)
        
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
