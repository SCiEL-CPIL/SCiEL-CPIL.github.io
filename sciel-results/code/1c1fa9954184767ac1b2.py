import time
import random
import heapq
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        # Try multiple restarts with different random seeds and density parameters
        for restart in range(200):
            if time.time() > deadline:
                break
                
            random.seed(restart * 98765 + 111)
            
            # Initialize graph with high probability to ensure high degrees
            # We need min_deg >= target_n - 39.
            # Let's aim for average degree slightly higher to have slack.
            # p = 0.5 is too low for n=47 if we need high min deg.
            # For n=47, min_deg needed is 8. p=0.5 gives avg deg 23.
            # But we will remove edges.
            # Let's start with a random graph with p=0.6 or 0.7.
            
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            
            # Build initial graph
            p_init = 0.6 + (restart % 10) * 0.01 # Vary density
            
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < p_init:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        deg[u] += 1
                        deg[v] += 1
                        edges.append([u, v])
            
            # Check if initial min degree is sufficient. If not, we might need to add edges?
            # No, we are removing. If min deg is already < needed, we fail immediately.
            # But with p=0.6, min deg is very likely >= 8 for n=47.
            if min(deg) < min_deg_needed:
                continue

            success = True
            # Iteratively remove edges to break C4s
            # We loop until no C4 is found or we get stuck.
            # To avoid infinite loops, we track that we are removing edges, so total edges decreases.
            # Max edges ~ n^2/2.
            
            while True:
                if time.time() > deadline:
                    success = False
                    break
                    
                # Find a C4
                c4_found = False
                c4_edges = []
                
                for u in range(target_n):
                    if c4_found:
                        break
                    for v in range(u + 1, target_n):
                        if c4_found:
                            break
                        # Check if u and v are non-adjacent
                        if (adj[u] >> v) & 1:
                            continue
                        
                        # Find common neighbors
                        common = adj[u] & adj[v]
                        
                        # Count bits
                        # If >= 2, we have a C4
                        # We need to identify two distinct neighbors w, x
                        # We don't need all, just two.
                        if common:
                            # Get first bit
                            w_bit = common & -common
                            w = w_bit.bit_length() - 1
                            common ^= w_bit
                            if common:
                                x_bit = common & -common
                                x = x_bit.bit_length() - 1
                                
                                # The C4 is u-w-v-x-u
                                # Edges: (u,w), (w,v), (v,x), (x,u)
                                c4_edges = [
                                    (min(u,w), max(u,w)),
                                    (min(w,v), max(w,v)),
                                    (min(v,x), max(v,x)),
                                    (min(x,u), max(x,u))
                                ]
                                c4_found = True
                                break
                            
                if not c4_found:
                    # No C4s left
                    break
                else:
                    # Try to remove an edge from c4_edges
                    removed = False
                    # Shuffle to randomize choice
                    random.shuffle(c4_edges)
                    for e in c4_edges:
                        u, v = e
                        # Check if we can remove (u,v)
                        # deg[u] and deg[v] must be > min_deg_needed
                        if deg[u] > min_deg_needed and deg[v] > min_deg_needed:
                            # Remove edge
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                            deg[u] -= 1
                            deg[v] -= 1
                            edges.remove([u, v])
                            removed = True
                            break
                    
                    if not removed:
                        # Stuck: C4 exists but no edge can be removed without violating min degree
                        success = False
                        break
            
            if success and min(deg) >= min_deg_needed:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                found = True
                break
            
        if best_n >= target_n:
            break
            
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
