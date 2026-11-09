import time
import random

def construct():
    n = 20
    adj = [0] * n
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
        
    def can_add(u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
            
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if popcount(adj[w] & adj[v]) >= 2:
                return False
            mask_u ^= lsb
            
        # Check neighbors of v
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

    def get_edges_list():
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

    def find_violation():
        for u in range(n):
            for v in range(u+1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return u, v
        return None

    def clean_graph():
        while True:
            viol = find_violation()
            if not viol:
                break
            u, v = viol
            common = adj[u] & adj[v]
            ws = []
            mask = common
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                ws.append(w)
                mask ^= lsb
            if not ws:
                break
            w = random.choice(ws)
            if random.random() < 0.5:
                remove_edge(u, w)
            else:
                remove_edge(v, w)

    def greedy_fill():
        added = False
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    if can_add(i, j):
                        add_edge(i, j)
                        added = True
        return added

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

    def try_swap(u1, v1, u2, v2):
        """Try removing edge (u1,v1) and adding edge (u2,v2)."""
        if u1 == u2 and v1 == v2:
            return False
        if u1 == u2 or v1 == u2 or u1 == v2 or v1 == v2:
            return False
        
        # Save state
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        
        # Check if we can add (u2, v2)
        can = True
        if (adj[u2] >> v2) & 1:
            can = False
        else:
            mask_u2 = adj[u2]
            while mask_u2 and can:
                lsb = mask_u2 & -mask_u2
                w = lsb.bit_length() - 1
                if popcount(adj[w] & adj[v2]) >= 2:
                    can = False
                mask_u2 ^= lsb
            
            if can:
                mask_v2 = adj[v2]
                while mask_v2 and can:
                    lsb = mask_v2 & -mask_v2
                    x = lsb.bit_length() - 1
                    if popcount(adj[u2] & adj[x]) >= 2:
                        can = False
                    mask_v2 ^= lsb
        
        if can:
            add_edge(u2, v2)
            return True
        else:
            # Restore
            add_edge(u1, v1)
            return False

    def try_remove_and_add(u1, v1, u2, v2):
        """Remove edge (u1,v1), then try to add edge (u2,v2)."""
        if (adj[u2] >> v2) & 1:
            return False
        
        # Save state
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        
        # Check if we can add (u2, v2)
        can = True
        mask_u2 = adj[u2]
        while mask_u2 and can:
            lsb = mask_u2 & -mask_u2
            w = lsb.bit_length() - 1
            if popcount(adj[w] & adj[v2]) >= 2:
                can = False
            mask_u2 ^= lsb
            
        if can:
            mask_v2 = adj[v2]
            while mask_v2 and can:
                lsb = mask_v2 & -mask_v2
                x = lsb.bit_length() - 1
                if popcount(adj[u2] & adj[x]) >= 2:
                    can = False
                mask_v2 ^= lsb
        
        if can:
            add_edge(u2, v2)
            return True
        else:
            # Restore
            add_edge(u1, v1)
            return False

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_adj = [0] * n
    best_count = 0
    
    # Multiple restarts with different initial densities
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        p = random.choice([0.3, 0.5, 0.7, 0.9])
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    add_edge(i, j)
        
        clean_graph()
        
        # Greedy fill
        greedy_fill()
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Local search with simulated annealing
        temperature = 1.0
        cooling = 0.995
        min_temp = 0.01
        
        while time.time() < deadline and temperature > min_temp:
            added = False
            
            # Try to add any possible edge
            for i in range(n):
                for j in range(i+1, n):
                    if not ((adj[i] >> j) & 1):
                        if can_add(i, j):
                            add_edge(i, j)
                            added = True
            
            current_count = count_edges()
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
            
            if not added:
                existing = get_existing_edges()
                
                if not existing:
                    break
                
                # Try swap moves
                move_made = False
                
                # With some probability, try a swap
                if random.random() < 0.3 and len(existing) > 1:
                    u1, v1 = random.choice(existing)
                    # Find a non-edge to potentially add
                    for i in range(n):
                        for j in range(i+1, n):
                            if not ((adj[i] >> j) & 1):
                                if try_swap(u1, v1, i, j):
                                    move_made = True
                                    break
                        if move_made:
                            break
                
                if not move_made:
                    # With some probability, try remove-and-add
                    if random.random() < 0.2 and existing:
                        u1, v1 = random.choice(existing)
                        for i in range(n):
                            for j in range(i+1, n):
                                if not ((adj[i] >> j) & 1):
                                    if try_remove_and_add(u1, v1, i, j):
                                        move_made = True
                                        break
                            if move_made:
                                break
                
                if not move_made:
                    # Random edge removal to escape local minimum
                    u, v = random.choice(existing)
                    remove_edge(u, v)
                
                # Accept based on simulated annealing
                new_count = count_edges()
                if new_count >= current_count or random.random() < pow(2.718281828, (new_count - current_count) / temperature):
                    pass  # Accept
                else:
                    # Revert - we need to undo the last move
                    # For simplicity, we'll just continue; the next iteration will try to recover
                    pass
                    
            temperature *= cooling
            
            if time.time() > deadline:
                break

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
