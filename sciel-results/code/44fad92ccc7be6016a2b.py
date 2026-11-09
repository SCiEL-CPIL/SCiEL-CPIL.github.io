import time
import random

def construct():
    n = 20
    adj = [0] * n
    common = [[0] * n for _ in range(n)]
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update common counts
        # For w in N(u) \ {v}, common[w][v] += 1
        mask_u = adj[u] & ~(1 << v)
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            common[w][v] += 1
            common[v][w] += 1
            mask_u ^= lsb
            
        # For x in N(v) \ {u}, common[u][x] += 1
        mask_v = adj[v] & ~(1 << u)
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            common[u][x] += 1
            common[x][u] += 1
            mask_v ^= lsb

    def remove_edge(u, v):
        # Reverse updates
        # For w in N(u) \ {v}, common[w][v] -= 1
        mask_u = adj[u] & ~(1 << v)
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            common[w][v] -= 1
            common[v][w] -= 1
            mask_u ^= lsb
            
        # For x in N(v) \ {u}, common[u][x] -= 1
        mask_v = adj[v] & ~(1 << u)
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            common[u][x] -= 1
            common[x][u] -= 1
            mask_v ^= lsb
            
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def can_add(u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
            
        # Check if adding (u,v) would make any common count exceed 2
        # For w in N(u), common[w][v] will increase by 1. Must be < 2 now.
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if common[w][v] >= 2:
                return False
            mask_u ^= lsb
            
        # For x in N(v), common[u][x] will increase by 1. Must be < 2 now.
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            if common[u][x] >= 2:
                return False
            mask_v ^= lsb
            
        return True

    def count_edges():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def get_edges_list(adj_state):
        edges = []
        for i in range(n):
            mask = adj_state[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_adj = [0] * n
    best_count = 0
    
    # We will run multiple restarts
    while time.time() < deadline:
        # Reset state
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        
        # Greedy Initialization
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if can_add(u, v):
                add_edge(u, v)
                
        current_count = count_edges()
        
        # Local Search
        improved = True
        local_stuck_count = 0
        while improved and time.time() < deadline:
            improved = False
            # Shuffle pairs for randomness in local search
            random.shuffle(pairs)
            for u, v in pairs:
                if not ((adj[u] >> v) & 1):
                    if can_add(u, v):
                        add_edge(u, v)
                        improved = True
                        current_count += 1
            
            if not improved:
                # Stuck, remove a random edge to escape
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
                            
                if existing:
                    u, v = random.choice(existing)
                    remove_edge(u, v)
                    current_count -= 1
                    local_stuck_count += 1
                    if local_stuck_count > 100:
                        # Too many stuck iterations, break to restart
                        break
                else:
                    break
                    
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
    return get_edges_list(best_adj)
