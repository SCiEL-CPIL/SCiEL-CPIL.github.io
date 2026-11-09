import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try n from 35 down to 18
    for n in range(35, 17, -1):
        if time.time() > deadline:
            break
            
        # Try a few restarts
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Random init
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= 1 << j
                        adj[j] |= 1 << i
            
            # Local Search
            for step in range(10000):
                if time.time() > deadline:
                    break
                    
                k4_count = count_k4(adj, n)
                
                if k4_count == 0:
                    # Check for Ind6
                    ind_set = find_max_independent_set(adj, n)
                    if len(ind_set) < 6:
                        edges = [[i, j] for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                        return {'n': n, 'edges': edges}
                    
                    # If Ind6 exists, add an edge to break it
                    # Pick two random vertices from ind_set
                    u, v = random.sample(ind_set, 2)
                    # Add edge
                    if not ((adj[u] >> v) & 1):
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                else:
                    # Minimize K4
                    # Pick a random edge and flip it if it reduces K4
                    # Or pick an edge involved in a K4 and remove it
                    # Removing an edge always reduces K4 count (or keeps same)
                    # Let's pick a random edge and flip
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v:
                        continue
                    
                    old_k4 = k4_count
                    # Flip
                    if (adj[u] >> v) & 1:
                        adj[u] ^= 1 << v
                        adj[v] ^= 1 << u
                    else:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                    
                    new_k4 = count_k4(adj, n)
                    if new_k4 > old_k4:
                        # Revert
                        if (adj[u] >> v) & 1:
                            adj[u] ^= 1 << v
                            adj[v] ^= 1 << u
                        else:
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
                    else:
                        k4_count = new_k4

    # Fallback to seed
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return {'n': 17, 'edges': edges}

def count_k4(adj, n):
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                mask = common
                while mask:
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    mask ^= lsb
                    total += (adj[w] & common).bit_count() // 2
    return total // 6

def find_max_independent_set(adj, n):
    # Simple greedy
    remaining = list(range(n))
    ind_set = []
    while remaining:
        # Pick vertex with lowest degree
        min_deg = float('inf')
        min_v = remaining[0]
        for v in remaining:
            deg = 0
            for u in remaining:
                if v != u and (adj[v] >> u) & 1:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_v = v
        
        ind_set.append(min_v)
        to_remove = set()
        to_remove.add(min_v)
        for u in remaining:
            if (adj[min_v] >> u) & 1:
                to_remove.add(u)
        remaining = [v for v in remaining if v not in to_remove]
        
    return ind_set
