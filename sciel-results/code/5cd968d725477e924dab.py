import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # We want to maximize n. Known best is 39.
    # We will try to find a valid graph for n=43, 42, 41, 40, 39...
    # The seed gives 30.
    
    def has_c4_conflict(adj, u, v):
        # Check if u and v have a common neighbor
        if len(adj[u]) > len(adj[v]):
            adj_u, adj_v = adj[v], adj[u]
        else:
            adj_u, adj_v = adj[u], adj[v]
        for w in adj_u:
            if w in adj_v:
                return True
        return False

    # Popcount helper
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count("1")

    def check_independence_number(adj, n, target):
        # Returns True if there is an independent set of size >= target
        if target <= 1:
            return True
        if n < target:
            return False
            
        adj_mask = [0] * n
        for u in range(n):
            mask = 0
            for v in adj[u]:
                mask |= (1 << v)
            adj_mask[u] = mask

        def dfs(available_mask, count):
            if count >= target:
                return True
            if available_mask == 0:
                return False
            
            # Pruning: if we can't reach target even if we pick all remaining
            if popcount(available_mask) + count < target:
                return False
            
            # Pick the lowest set bit
            lsb = available_mask & -available_mask
            u = lsb.bit_length() - 1
            
            # Option 1: Do not pick u
            if dfs(available_mask ^ lsb, count):
                return True
            
            # Option 2: Pick u
            # Remove u and its neighbors from available
            new_mask = available_mask & ~(adj_mask[u] | (1 << u))
            if dfs(new_mask, count + 1):
                return True
            
            return False
        
        initial_mask = (1 << n) - 1
        return dfs(initial_mask, 0)

    # Try n from 43 down to 30
    for n in range(43, 29, -1):
        # Try multiple random restarts for this n
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            adj = [set() for _ in range(n)]
            
            # Generate all potential edges and shuffle
            pairs = [(i, j) for i in range(n) for j in range(i)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if not has_c4_conflict(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Check independence number
            # If alpha < 11, we have a valid graph
            if not check_independence_number(adj, n, 11):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
        if time.time() > deadline:
            break

    # Fallback to seed solution if nothing better found
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
