import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try n=21 first
    result = search_graph(21, deadline)
    if result:
        return result
        
    # Try n=20
    result = search_graph(20, deadline)
    if result:
        return result
        
    # Fallback n=5
    return {'n': 5, 'edges': []}

def search_graph(n, deadline):
    # Precompute masks of size 6 for fast checking
    # We check size 6 primarily. If 0 bad size-6, we do full check.
    # Generating all masks of size 6 for n=21
    masks_6 = []
    # Efficient generation of combinations?
    # Just iterate range(1<<n) and filter? 2^21 = 2M. 
    # 2M iterations in Python takes ~0.2-0.5s. Acceptable.
    for mask in range(1 << n):
        if mask.bit_count() == 6:
            masks_6.append(mask)
            
    # Random search with local optimization
    best_graph = None
    best_score = float('inf')
    
    for _ in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        # Random init
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Local search
        current_score = count_bad_size6(n, adj, masks_6)
        
        for step in range(200):
            if time.time() > deadline:
                break
                
            if current_score == 0:
                # Check full validity
                if is_valid(n, adj):
                    edges = get_edges(n, adj)
                    return {'n': n, 'edges': edges}
                
            # Pick random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            # Flip
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            new_score = count_bad_size6(n, adj, masks_6)
            
            if new_score <= current_score:
                current_score = new_score
            else:
                # Revert
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
    return None

def count_bad_size6(n, adj, masks_6):
    count = 0
    for mask in masks_6:
        if is_regular_mask(n, adj, mask):
            count += 1
    return count

def is_regular_mask(n, adj, mask):
    # Get vertices in mask
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
        
    # Check degrees
    degs = []
    for v in verts:
        # Degree in mask
        d = (adj[v] & mask).bit_count()
        degs.append(d)
        
    # Check if all same
    first = degs[0]
    for d in degs[1:]:
        if d != first:
            return False
    return True

def is_valid(n, adj):
    # Check all subsets size >= 6
    # To speed up, we can iterate all masks and check size
    for mask in range(1 << n):
        k = mask.bit_count()
        if k >= 6:
            if is_regular_mask(n, adj, mask):
                return False
    return True

def get_edges(n, adj):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges
