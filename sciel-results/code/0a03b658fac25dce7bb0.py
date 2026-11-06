# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 110.0  # Safety margin
    
    random.seed(42)
    
    edges_set = set()
    # Bipartite K_20,20
    for a in range(20):
        for b in range(20, 40):
            edges_set.add((a, b))
    
    # Add one intra-part edge
    edges_set.add((0, 1))
    
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges_set:
        adj[u, v] = 1
        adj[v, u] = 1
        
    # Compute initial tri
    tri = (adj @ adj).astype(np.int32)
    
    # Helper to get max book size
    def get_max_book(adj, tri):
        # tri[u, v] is number of common neighbors
        # We only care about edges (adj[u, v] == 1)
        # Since adj is symmetric, we can just take max of tri where adj is 1
        # Note: tri[u, u] is degree, but adj[u, u] is 0, so it's excluded.
        return tri[adj == 1].max()

    current_max_book = get_max_book(adj, tri)
    best_adj = adj.copy()
    best_tri = tri.copy()
    best_max_book = current_max_book
    
    # Local search
    # We will use a temperature-like variable for acceptance
    # Start with strict improvement, then allow plateaus
    
    no_improve = 0
    max_no_improve = 200
    temp = 0.0 # 0 means only improvements
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Perturb: accept a worse move to escape
            # Try to find a move that reduces the max, or if not, accept a random one
            # For simplicity, just do a few random swaps and accept the best among them
            # or accept any if it's not too bad?
            # Let's try: pick 5 random swaps, evaluate, take the best one.
            # If the best is worse than current, accept it with prob 0.5?
            # Or just accept it to jump out.
            
            best_candidate_adj = adj.copy()
            best_candidate_tri = tri.copy()
            best_candidate_max = current_max_book
            
            for _ in range(5):
                # Random swap
                while True:
                    u1 = random.randint(0, n-1)
                    v1 = random.randint(0, n-1)
                    if u1 == v1: continue
                    if adj[u1, v1]: break
                
                while True:
                    u2 = random.randint(0, n-1)
                    v2 = random.randint(0, n-1)
                    if u2 == v2: continue
                    if not adj[u2, v2]: break
                
                # Apply swap
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                tri = (adj @ adj).astype(np.int32)
                new_max = get_max_book(adj, tri)
                
                # Revert
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                
                if new_max < best_candidate_max:
                    best_candidate_max = new_max
                    best_candidate_adj = adj.copy() # Wait, adj was reverted. Need to re-apply or store state before revert.
                    # This is inefficient. Let's just pick one random swap and accept it if it's not too bad.
                    pass
            
            # Simpler Perturb: Just do one random swap and accept it regardless (or if it's not terrible)
            # Let's accept any move if no_improve > max_no_improve
            while True:
                u1 = random.randint(0, n-1)
                v1 = random.randint(0, n-1)
                if u1 == v1: continue
                if adj[u1, v1]: break
            
            while True:
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2: continue
                if not adj[u2, v2]: break
            
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            
            tri = (adj @ adj).astype(np.int32)
            current_max_book = get_max_book(adj, tri)
            
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_adj = adj.copy()
                best_tri = tri.copy()
                no_improve = 0
            else:
                no_improve = 0 # Reset counter to allow further random walks
            continue
        
        # Normal step
        while True:
            u1 = random.randint(0, n-1)
            v1 = random.randint(0, n-1)
            if u1 == v1: continue
            if adj[u1, v1]: break
        
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2: continue
            if not adj[u2, v2]: break
        
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        tri = (adj @ adj).astype(np.int32)
        new_max = get_max_book(adj, tri)
        
        if new_max < current_max_book:
            current_max_book = new_max
            if new_max < best_max_book:
                best_max_book = new_max
                best_adj = adj.copy()
                best_tri = tri.copy()
            no_improve = 0
        elif new_max == current_max_book:
            # Accept with some probability to explore
            if random.random() < 0.1:
                current_max_book = new_max
                no_improve += 1
            else:
                # Revert
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                no_improve += 1
        else:
            # Revert
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            no_improve += 1

    edges_list = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v]:
                edges_list.append([u, v])
    
    return edges_list

# EVOLVE-BLOCK-END
