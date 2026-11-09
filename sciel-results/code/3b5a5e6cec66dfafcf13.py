import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_n = 0
    best_edges = []
    
    # Try n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_G = target_n - 39
        max_deg_bar = 38
        
        found = False
        # Try a few random seeds
        for seed in range(20):
            if time.time() > deadline:
                break
                
            random.seed(seed)
            
            # Initialize bar_G as empty
            adj_bar = [0] * target_n
            deg_bar = [0] * target_n
            
            # We want to build bar_G such that:
            # 1. deg_bar[u] <= 38
            # 2. For all u != v, |N_bar(u) U N_bar(v)| >= target_n - 3
            
            # Start with empty graph
            # Iteratively add edges to fix violations
            
            # Helper to get neighbors list from mask
            def get_neighbors(mask):
                neigh = []
                m = mask
                while m:
                    lsb = m & -m
                    neigh.append(lsb.bit_length() - 1)
                    m ^= lsb
                return neigh

            # Main loop
            max_iter = 5000
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Find a violation
                violation_found = False
                # To speed up, we can check a random subset or all. n=47 is small.
                # Shuffle pairs to avoid bias
                pairs = [(u, v) for u in range(target_n) for v in range(u+1, target_n)]
                random.shuffle(pairs)
                
                for u, v in pairs:
                    # Calculate union size
                    # Union = deg[u] + deg[v] - (adj[u] & adj[v]).bit_count()
                    common = adj_bar[u] & adj_bar[v]
                    common_count = common.bit_count()
                    union_size = deg_bar[u] + deg_bar[v] - common_count
                    
                    if union_size < target_n - 3:
                        violation_found = True
                        
                        # Try to fix by adding an edge
                        # Option 1: Add (u, w) where w not in N(v) and deg[u] < 38
                        # Option 2: Add (v, w) where w not in N(u) and deg[v] < 38
                        
                        fixed = False
                        
                        # Try adding to u
                        if deg_bar[u] < max_deg_bar:
                            # Find w not in N(v) and w != u, w not in N(u) (to be a new edge)
                            # Actually w just needs to not be connected to u yet.
                            # And w not in N(v).
                            candidates = []
                            for w in range(target_n):
                                if w == u or w == v:
                                    continue
                                # Check if w is not neighbor of v
                                if not (adj_bar[v] & (1 << w)):
                                    # Check if w is not neighbor of u (edge doesn't exist)
                                    if not (adj_bar[u] & (1 << w)):
                                        candidates.append(w)
                            
                            if candidates:
                                w = random.choice(candidates)
                                # Add edge (u, w)
                                adj_bar[u] |= 1 << w
                                adj_bar[w] |= 1 << u
                                deg_bar[u] += 1
                                deg_bar[w] += 1
                                fixed = True
                                break # Restart pair check? Or just continue? 
                                # Breaking the pair loop and restarting the outer loop is safer to re-evaluate all.
                                # But it's expensive. Let's just mark fixed and break inner loop.
                        
                        if not fixed and deg_bar[v] < max_deg_bar:
                            candidates = []
                            for w in range(target_n):
                                if w == u or w == v:
                                    continue
                                if not (adj_bar[u] & (1 << w)):
                                    if not (adj_bar[v] & (1 << w)):
                                        candidates.append(w)
                            
                            if candidates:
                                w = random.choice(candidates)
                                adj_bar[v] |= 1 << w
                                adj_bar[w] |= 1 << v
                                deg_bar[v] += 1
                                deg_bar[w] += 1
                                fixed = True
                                break
                                
                        if not fixed:
                            # Stuck? Try removing an edge to break symmetry?
                            # This is complex. For now, just break and try next seed.
                            break 
                            
                if not violation_found:
                    # Check if valid
                    # Re-verify all pairs
                    valid = True
                    for u in range(target_n):
                        for v in range(u+1, target_n):
                            common = adj_bar[u] & adj_bar[v]
                            common_count = common.bit_count()
                            union_size = deg_bar[u] + deg_bar[v] - common_count
                            if union_size < target_n - 3:
                                valid = False
                                break
                        if not valid:
                            break
                            
                    if valid:
                        # Construct G edges
                        edges_G = []
                        for u in range(target_n):
                            for v in range(u+1, target_n):
                                if not (adj_bar[u] & (1 << v)):
                                    edges_G.append([u, v])
                                    
                        # Verify min degree of G
                        # deg_G(u) = target_n - 1 - deg_bar[u]
                        min_deg = min(target_n - 1 - d for d in deg_bar)
                        if min_deg >= min_deg_G:
                            if target_n > best_n:
                                best_n = target_n
                                best_edges = edges_G
                            found = True
                            break
                            
            if found:
                break
                
    if best_n == 0:
        return {'n': 1, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
