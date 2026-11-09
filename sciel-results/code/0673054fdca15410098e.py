import time
import random
import sys

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
        max_int = 79 - target_n
        
        # We construct the complement graph bar_G
        # bar_G must have max degree <= 38
        # and for all u != v, |N_bar(u) & N_bar(v)| <= max_int
        
        found = False
        # Try a few random seeds
        for seed in range(50):
            if time.time() > deadline:
                break
                
            random.seed(seed)
            
            # Initialize bar_G as empty
            adj_bar = [0] * target_n
            edges_bar = []
            
            # Generate random edges
            # Probability p such that expected degree is around 38
            # p = 38 / (target_n - 1)
            p = 38.0 / (target_n - 1)
            if p > 0.9: p = 0.9
            
            all_edges = [(u, v) for u in range(target_n) for v in range(u+1, target_n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if random.random() < p:
                    adj_bar[u] |= 1 << v
                    adj_bar[v] |= 1 << u
                    edges_bar.append((u, v))
            
            # Repair phase
            # We need to ensure max degree <= 38 and max intersection <= max_int
            # Iterate until no violations or max iterations
            for _ in range(2000):
                if time.time() > deadline:
                    break
                    
                # Check degrees
                deg_violation = False
                for u in range(target_n):
                    deg = bin(adj_bar[u]).count('1')
                    if deg > max_deg_bar:
                        # Remove a random edge from u
                        # Find neighbors
                        mask = adj_bar[u]
                        # Pick a random set bit
                        # Simple way: convert to list, pick random, clear bit
                        # But bit iteration is faster
                        # Let's just pick a random neighbor
                        # We can store edges in a set for O(1) removal? 
                        # Or just find a bit.
                        # To remove edge (u, v):
                        # adj_bar[u] ^= 1 << v
                        # adj_bar[v] ^= 1 << u
                        # edges_bar.remove((u, v)) is slow. 
                        # We can just rebuild edges_bar at the end.
                        
                        # Pick random neighbor
                        # Get list of set bits
                        neighbors = []
                        m = mask
                        while m:
                            lsb = m & -m
                            neighbors.append(lsb.bit_length() - 1)
                            m ^= lsb
                        if neighbors:
                            v = random.choice(neighbors)
                            adj_bar[u] ^= 1 << v
                            adj_bar[v] ^= 1 << u
                            deg_violation = True
                            break
                            
                if deg_violation:
                    continue
                    
                # Check intersections
                int_violation = False
                # Sample some pairs or check all?
                # n=47, pairs = 47*46/2 = 1081.
                # Checking all is fast.
                for u in range(target_n):
                    for w in range(u+1, target_n):
                        common = adj_bar[u] & adj_bar[w]
                        count = bin(common).count('1')
                        if count > max_int:
                            # Remove an edge to reduce intersection
                            # Edges incident to u or w that are in common?
                            # If we remove (u, x) where x is in common, 
                            # does it reduce |N(u) & N(w)|?
                            # x is in N(u) and N(w).
                            # If we remove (u, x), x is no longer in N(u).
                            # So x is no longer in N(u) & N(w).
                            # So yes.
                            # Pick a random x from common
                            if common:
                                m = common
                                # Pick random bit
                                # Get list
                                commons = []
                                while m:
                                    lsb = m & -m
                                    commons.append(lsb.bit_length() - 1)
                                    m ^= lsb
                                x = random.choice(commons)
                                # Remove edge (u, x)
                                adj_bar[u] ^= 1 << x
                                adj_bar[x] ^= 1 << u
                                int_violation = True
                                break
                    if int_violation:
                        break
                        
                if not int_violation and not deg_violation:
                    # Valid graph found
                    # Reconstruct edges_bar from adj_bar
                    edges_bar_final = []
                    for u in range(target_n):
                        for v in range(u+1, target_n):
                            if adj_bar[u] & (1 << v):
                                edges_bar_final.append([u, v])
                    
                    # The task asks for G, not bar_G.
                    # G is the complement of bar_G.
                    # So edges of G are all pairs NOT in edges_bar_final.
                    edges_G = []
                    for u in range(target_n):
                        for v in range(u+1, target_n):
                            if not (adj_bar[u] & (1 << v)):
                                edges_G.append([u, v])
                    
                    # Verify min degree of G
                    # deg_G(u) = target_n - 1 - deg_bar(u)
                    min_deg = min(target_n - 1 - bin(adj_bar[u]).count('1') for u in range(target_n))
                    if min_deg >= min_deg_G:
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = edges_G
                        found = True
                        break
                        
            if found:
                break
                
    if best_n == 0:
        # Fallback to trivial
        return {'n': 1, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}

