import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_n = 0
    best_edges = []

    # Try n=47 first, then 46, then 45...
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        if min_deg_needed < 0:
            min_deg_needed = 0
            
        found = False
        
        # Number of restarts depends on time available
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart * 100 + target_n)
            
            # Initialize with complete graph
            N = [set(range(target_n)) for _ in range(target_n)]
            for i in range(target_n):
                N[i].remove(i)
            
            deg = [target_n - 1] * target_n
            
            # common[u][v] = |N(u) cap N(v)|
            common = [[0] * target_n for _ in range(target_n)]
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    c = len(N[u] & N[v])
                    common[u][v] = c
                    common[v][u] = c

            def remove_edge(u, v):
                N[u].remove(v)
                N[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
                
                # Update common
                for w in N[u]:
                    if w != v:
                        common[u][w] -= 1
                        common[w][u] -= 1
                        common[v][w] -= 1
                        common[w][v] -= 1
                for w in N[v]:
                    if w != u:
                        common[u][w] -= 1
                        common[w][u] -= 1
                        common[v][w] -= 1
                        common[w][v] -= 1
                common[u][v] = 0
                common[v][u] = 0

            def add_edge(u, v):
                N[u].add(v)
                N[v].add(u)
                deg[u] += 1
                deg[v] += 1
                
                # Update common
                for w in N[u]:
                    if w != v:
                        common[u][w] += 1
                        common[w][u] += 1
                        common[v][w] += 1
                        common[w][v] += 1
                for w in N[v]:
                    if w != u:
                        common[u][w] += 1
                        common[w][u] += 1
                        common[v][w] += 1
                        common[w][v] += 1
                common[u][v] = 1
                common[v][u] = 1

            def can_add(u, v):
                if v in N[u]:
                    return False
                if common[u][v] > 1:
                    return False
                for w in N[v]:
                    if common[u][w] > 0:
                        return False
                for w in N[u]:
                    if common[v][w] > 0:
                        return False
                return True

            success = True
            
            # Phase 1: Remove edges to break all C4s
            # We want to maintain min degree >= min_deg_needed
            max_removals = 1000
            removals = 0
            
            while removals < max_removals:
                if time.time() > deadline:
                    success = False
                    break
                
                # Find a C4
                c4_found = False
                c4_edges = []
                
                for u in range(target_n):
                    if not c4_found:
                        for v in range(u + 1, target_n):
                            if common[u][v] >= 2:
                                # Get two common neighbors
                                common_neighbors = list(N[u] & N[v])
                                # We need at least 2
                                w1, w2 = random.sample(common_neighbors, 2)
                                # The C4 is u-w1-v-w2-u
                                # Edges: (u,w1), (w1,v), (v,w2), (w2,u)
                                c4_edges = [(u, w1), (w1, v), (v, w2), (w2, u)]
                                c4_found = True
                                break
                
                if not c4_found:
                    break # No C4s left
                
                # Choose an edge to remove
                # Prefer edges where both endpoints have degree > min_deg_needed
                valid_edges = []
                for e in c4_edges:
                    x, y = e
                    if deg[x] > min_deg_needed and deg[y] > min_deg_needed:
                        valid_edges.append(e)
                
                if not valid_edges:
                    # All edges in this C4 are critical. 
                    # Try to find another C4 or fail
                    # For simplicity, pick a random edge and hope it doesn't break min deg? 
                    # No, that would violate the condition.
                    # If we are stuck, we fail this restart.
                    success = False
                    break
                
                # Pick the edge with highest sum of degrees (to preserve low degree vertices)
                e_to_remove = max(valid_edges, key=lambda e: deg[e[0]] + deg[e[1]])
                remove_edge(e_to_remove[0], e_to_remove[1])
                removals += 1
                
            if not success:
                continue

            # Phase 2: Add edges to increase min degree if needed
            if min(deg) < min_deg_needed:
                # Greedy addition
                for _ in range(1000):
                    if time.time() > deadline:
                        success = False
                        break
                    
                    low_vertices = [i for i in range(target_n) if deg[i] < min_deg_needed]
                    if not low_vertices:
                        break
                        
                    u = random.choice(low_vertices)
                    candidates = []
                    for v in range(target_n):
                        if v == u:
                            continue
                        if can_add(u, v):
                            candidates.append(v)
                            
                    if not candidates:
                        # Stuck in addition phase
                        # Try to remove a random edge to unblock?
                        # Or just fail
                        success = False
                        break
                        
                    v = random.choice(candidates)
                    add_edge(u, v)
                    
            if not success:
                continue

            # Final Verification
            if min(deg) >= min_deg_needed:
                c4_free = True
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if common[i][j] > 1:
                            c4_free = False
                            break
                    if not c4_free:
                        break
                
                if c4_free:
                    final_edges = []
                    for i in range(target_n):
                        for j in N[i]:
                            if i < j:
                                final_edges.append([i, j])
                    
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = final_edges
                    found = True
                    break # Break restart loop
        
        if best_n == 47:
            break
        if found:
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: n=39, empty graph is valid (min deg 0 >= 0)
        return {'n': 39, 'edges': []}
