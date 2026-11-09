import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Fallback to seed
    def get_seed():
        n = 26
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
        return {'n': n, 'edges': edges}
    
    best_result = get_seed()
    
    # Try to find larger n
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
            
        found = try_find_graph(n, deadline)
        if found:
            return found
            
    return best_result

def try_find_graph(n, deadline):
    # Try random restarts
    for attempt in range(50):
        if time.time() > deadline:
            return None
            
        # Generate a random triangle-free graph
        # Start with a random 4-regular graph
        adj = generate_random_triangle_free(n, 4)
        if adj is None:
            continue
            
        # Local search to minimize alpha
        adj = local_search_alpha(n, adj, deadline)
        
        # Verify with B&B
        if verify_alpha_less_than_10(n, adj, deadline):
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return None

def generate_random_triangle_free(n, degree):
    # Simple random regular graph generation, check triangle-free
    # Retry logic inside
    for _ in range(10):
        adj = np.zeros((n, n), dtype=bool)
        # Random permutation of neighbors
        # This is a bit complex to do properly for regular graphs without libraries.
        # Alternative: Start with empty, add random edges, check triangle-free.
        # But we want specific degree.
        
        # Let's use a simpler heuristic: Random graph G(n, p) with p small, then add edges to reach degree?
        # Or just random edges, ensure no triangle.
        
        adj = np.zeros((n, n), dtype=bool)
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.15: # Sparse
                    # Check if adding (i,j) creates a triangle
                    if not np.any(adj[i] & adj[j]):
                        adj[i][j] = True
                        adj[j][i] = True
                        edges.append((i,j))
                        
        # Check if it's connected? Not required, but good.
        # Check alpha?
        return adj

def local_search_alpha(n, adj, deadline):
    # Try to add edges to reduce alpha
    # Find a large independent set
    # If |I| >= 10, try to add an edge (u,v) where u,v in I? No, u,v in I are not adjacent.
    # Adding (u,v) breaks I.
    # Constraint: N(u) cap N(v) is empty.
    
    current_alpha = estimate_alpha(n, adj)
    if current_alpha < 10:
        return adj
        
    for _ in range(200):
        if time.time() > deadline:
            return adj
            
        I = find_max_independent_set_greedy(n, adj)
        if len(I) < 10:
            return adj
            
        # Pick two vertices u, v from I such that they are not adjacent and no common neighbor
        # Randomly sample pairs from I
        candidates = []
        for idx_u in range(len(I)):
            u = I[idx_u]
            for idx_v in range(idx_u+1, len(I)):
                v = I[idx_v]
                if not adj[u][v] and not np.any(adj[u] & adj[v]):
                    candidates.append((u, v))
                    
        if not candidates:
            # Try to add edges between vertices not in I?
            # Or restart
            break
            
        u, v = random.choice(candidates)
        adj[u][v] = True
        adj[v][u] = True
        
    return adj

def estimate_alpha(n, adj):
    # Greedy
    remaining = np.ones(n, dtype=bool)
    count = 0
    while np.any(remaining):
        # Pick vertex with min degree in remaining
        degs = adj[remaining].sum(axis=1)
        min_idx = np.argmin(degs)
        # Map back to original index
        verts = np.where(remaining)[0]
        u = verts[min_idx]
        count += 1
        remaining[u] = False
        remaining[adj[u]] = False
    return count

def find_max_independent_set_greedy(n, adj):
    # Greedy maximal independent set
    remaining = np.ones(n, dtype=bool)
    I = []
    while np.any(remaining):
        # Pick random vertex from remaining? Or min degree?
        # Min degree is better for finding large sets
        degs = adj[remaining].sum(axis=1)
        min_idx = np.argmin(degs)
        verts = np.where(remaining)[0]
        u = verts[min_idx]
        I.append(u)
        remaining[u] = False
        remaining[adj[u]] = False
    return I

def verify_alpha_less_than_10(n, adj, deadline):
    # Simple B&B
    # adj is numpy bool matrix
    best = 0
    start_time = time.time()
    
    def bnb(remaining_mask, current_count):
        nonlocal best
        if time.time() > deadline - 5:
            return False # Timeout
            
        if current_count + np.sum(remaining_mask) <= best:
            return True
            
        if not np.any(remaining_mask):
            if current_count > best:
                best = current_count
            return best < 10
            
        # Heuristic: pick vertex with max degree in remaining
        degs = adj[remaining_mask].sum(axis=1)
        # degs is vector of size n, but only valid for remaining
        # Find max
        max_deg = -1
        u = -1
        verts = np.where(remaining_mask)[0]
        for v in verts:
            if degs[v] > max_deg:
                max_deg = degs[v]
                u = v
                
        if u == -1:
            return best < 10
            
        # Branch 1: Include u
        new_remaining = remaining_mask.copy()
        new_remaining[u] = False
        new_remaining[adj[u]] = False
        
        if bnb(new_remaining, current_count + 1):
            return True
            
        # Branch 2: Exclude u
        new_remaining = remaining_mask.copy()
        new_remaining[u] = False
        
        if bnb(new_remaining, current_count):
            return True
            
        return best < 10

    initial_mask = np.ones(n, dtype=bool)
    success = bnb(initial_mask, 0)
    return success

