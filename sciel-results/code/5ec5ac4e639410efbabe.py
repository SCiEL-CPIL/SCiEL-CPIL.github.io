import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try n=21 first
    for n in [21, 20]:
        if time.time() > deadline:
            break
        graph = find_graph(n, deadline)
        if graph is not None:
            return {'n': n, 'edges': graph}
            
    # Fallback to a known valid small graph if all else fails
    # e.g. empty graph on 5 vertices
    return {'n': 5, 'edges': []}

def find_graph(n, deadline):
    # Precompute subsets of size 6
    subsets_6 = []
    # Generate combinations of 6
    import itertools
    for comb in itertools.combinations(range(n), 6):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        subsets_6.append(mask)
        
    # Larger subsets can be checked later or added dynamically
    # Let's start with just size 6 for speed, then expand
    
    best_edges = None
    best_count = float('inf')
    
    for restart in range(100):
        if time.time() > deadline:
            break
        # Random init
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        count = count_bad(adj, n, subsets_6)
        if count == 0:
            # Check larger subsets
            if check_larger(adj, n):
                return edges_from_adj(adj, n)
        
        # Local search
        for step in range(500):
            if time.time() > deadline:
                break
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            
            # Flip edge
            mask_uv = (1 << u) | (1 << v)
            # Check if edge exists
            if adj[u] & (1 << v):
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            new_count = count_bad(adj, n, subsets_6)
            
            # Simple hill climbing: accept if better or equal (with prob)
            if new_count < count or (new_count == count and random.random() < 0.1):
                count = new_count
                if count == 0:
                    if check_larger(adj, n):
                        return edges_from_adj(adj, n)
            else:
                # Revert
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
    return None

def count_bad(adj, n, subsets):
    count = 0
    for mask in subsets:
        # Check if induced subgraph on mask is regular
        # Get vertices in mask
        verts = [i for i in range(n) if (mask >> i) & 1]
        if not verts: continue
        degs = []
        for v in verts:
            degs.append((adj[v] & mask).bit_count())
        if len(set(degs)) == 1:
            count += 1
    return count

def check_larger(adj, n):
    # Check subsets of size 7..n
    import itertools
    for k in range(7, n+1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            verts = [i for i in range(n) if (mask >> i) & 1]
            degs = [(adj[v] & mask).bit_count() for v in verts]
            if len(set(degs)) == 1:
                return False
    return True

def edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges
