import time
import random
from math import log, exp, sqrt

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    n = 20
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    # cc[u][v] is the number of common neighbors of u and v
    cc = [[0] * n for _ in range(n)]
    
    def can_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        if cc[u][v] > 2:
            return False
        # Check neighbors of v
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if cc[u][w] > 1:
                return False
            mask_v ^= lsb
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if cc[v][w] > 1:
                return False
            mask_u ^= lsb
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update cc
        old_mask_v = adj[v] ^ (1 << u)
        m = old_mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] += 1
            cc[w][u] += 1
            m ^= lsb
            
        old_mask_u = adj[u] ^ (1 << v)
        m = old_mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] += 1
            cc[w][v] += 1
            m ^= lsb

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update cc
        m = adj[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] -= 1
            cc[w][u] -= 1
            m ^= lsb
            
        m = adj[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] -= 1
            cc[w][v] -= 1
            m ^= lsb

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def edge_count():
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def greedy_fill(order=None):
        if order is None:
            order = list(range(n))
            random.shuffle(order)
        # Try to add all possible edges in random order
        possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(possible)
        for u, v in possible:
            if can_add(u, v):
                add_edge(u, v)

    def local_search():
        """Intensive local search: try all possible moves and accept improvements"""
        improved = True
        while improved:
            improved = False
            # Try adding edges
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1) and can_add(u, v):
                        add_edge(u, v)
                        improved = True
            # Try removing edges if it allows adding more (2-opt style)
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        # Try removing this edge
                        remove_edge(u, v)
                        # Try to add as many edges as possible
                        added = 0
                        for a in range(n):
                            for b in range(a+1, n):
                                if not ((adj[a] >> b) & 1) and can_add(a, b):
                                    add_edge(a, b)
                                    added += 1
                        if added > 1:
                            # Net gain, keep the change
                            improved = True
                        else:
                            # Revert
                            remove_edge(u, v) if False else None
                            # Actually we need to undo the additions
                            # This is complex, let's just not do this
                            # Re-add the original edge
                            add_edge(u, v)
                            # Remove the added edges - we need to track them
                            # This approach is flawed, let's skip 2-opt for now
                            break
        return improved

    # Initialize with multiple random restarts
    random.seed(42)
    best_edges = []
    best_count = 0
    
    # Multiple greedy starts with different random seeds
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Reset
        adj = [0] * n
        cc = [[0] * n for _ in range(n)]
        
        # Random greedy construction
        possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(possible)
        for u, v in possible:
            if can_add(u, v):
                add_edge(u, v)
        
        # Local search: try to add any edge that can be added
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1) and can_add(u, v):
                        add_edge(u, v)
                        improved = True
        
        count = edge_count()
        if count > best_count:
            best_count = count
            best_edges = get_edges()
    
    # Simulated annealing for the best solution
    # Reconstruct best
    adj = [0] * n
    cc = [[0] * n for _ in range(n)]
    for u, v in best_edges:
        add_edge(u, v)
    
    T = 1.0
    cooling = 0.99995
    min_T = 0.01
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() < deadline - 5:
        # Pick a random move
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        if (adj[u] >> v) & 1:
            # Edge exists, try to remove
            remove_edge(u, v)
            # Try to add a random other edge
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 != v2 and not ((adj[u2] >> v2) & 1):
                if can_add(u2, v2):
                    add_edge(u2, v2)
                    # Net change 0 (swap)
                else:
                    add_edge(u, v)
            else:
                add_edge(u, v)
        else:
            # Edge does not exist, try to add
            if can_add(u, v):
                add_edge(u, v)
            else:
                # Try a swap: remove a random existing edge, add this one
                # Find a random existing edge
                existing = []
                for a in range(n):
                    for b in range(a+1, n):
                        if (adj[a] >> b) & 1:
                            existing.append((a, b))
                if existing:
                    a, b = random.choice(existing)
                    remove_edge(a, b)
                    if can_add(u, v):
                        add_edge(u, v)
                    else:
                        add_edge(a, b)
        
        T *= cooling
        if T < min_T:
            T = min_T
        
        # Check if we improved
        count = edge_count()
        if count > best_count:
            best_count = count
            best_edges = get_edges()
        
        # Occasional greedy fill
        if random.random() < 0.01:
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1) and can_add(u, v):
                        add_edge(u, v)
    
    # Final greedy pass
    for u in range(n):
        for v in range(u+1, n):
            if not ((adj[u] >> v) & 1) and can_add(u, v):
                add_edge(u, v)
    
    edges = get_edges()
    if len(edges) > best_count:
        best_edges = edges
        best_count = len(edges)
        
    return best_edges
