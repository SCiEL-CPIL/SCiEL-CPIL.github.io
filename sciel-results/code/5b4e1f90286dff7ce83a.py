import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
            
        # Run multiple attempts for this n
        found = False
        attempts = 0
        max_attempts = 100
        
        while not found and time.time() < deadline:
            attempts += 1
            # Random seed for this attempt
            random.seed(attempts * n + 12345)
            
            # adj[i] is a set of neighbors
            adj = [set() for _ in range(n)]
            # degrees
            deg = [0] * n
            edges = []
            
            # Precompute all pairs as a list
            all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            
            # Strategy:
            # While min(deg) < target_deg:
            #   Find candidate edges (u,v) not in adj, no C4.
            #   Select best candidate.
            
            while min(deg) < target_deg:
                if time.time() > deadline:
                    break
                
                best_edge = None
                best_score = float('inf')
                
                # Iterate all pairs
                for u in range(n):
                    for v in range(u + 1, n):
                        if v in adj[u]:
                            continue
                        
                        # Check C4
                        # Path u-x-y-v exists?
                        # For x in adj[u]:
                        #   if adj[x] & adj[v]: C4
                        c4_exists = False
                        for x in adj[u]:
                            if adj[x] & adj[v]:
                                c4_exists = True
                                break
                        if c4_exists:
                            continue
                        
                        # Score: prioritize increasing min degree
                        # We want to add edge to low degree vertices
                        score = min(deg[u], deg[v])
                        # Tie break: sum of degrees? or random
                        if score < best_score:
                            best_score = score
                            best_edge = (u, v)
                        elif score == best_score:
                            # Random tie break
                            if random.random() < 0.3:
                                best_edge = (u, v)
                
                if best_edge is None:
                    break # No more edges can be added
                
                u, v = best_edge
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
            
            if min(deg) >= target_deg:
                found = True
                if n > best_n:
                    best_n = n
                    best_edges = [e[:] for e in edges]
        
        if found:
            # We found a valid graph for this n. Since we go 47->40, this is the max.
            # We can break out of the n loop if we want, but maybe check time.
            # The task says "Maximize n". So first found is best.
            break
            
    return {'n': best_n, 'edges': best_edges}
