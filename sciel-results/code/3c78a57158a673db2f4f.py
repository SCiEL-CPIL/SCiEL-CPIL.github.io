import time
import random
import heapq
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    # We try to find a graph for n=47, 46, ...
    # Known R(C4, K1,39) > n implies existence of C4-free graph with min deg >= n-39
    # We maximize n.
    
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        # Try multiple restarts
        # For n=47, d_min=8. For n=46, d_min=7.
        # Number of restarts can be high because the check is fast.
        num_restarts = 500 if target_n <= 45 else 200
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart * 7919 + target_n * 131)
            
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            
            # List of all potential edges
            all_edges = [(u, v) for u in range(target_n) for v in range(u + 1, target_n)]
            random.shuffle(all_edges)
            
            # We will iterate through the shuffled list, but we need to be smart.
            # Actually, we can just loop and pick the best valid edge each time.
            # But scanning all edges each time is O(M).
            # M ~ 1000.
            
            success = True
            while True:
                if time.time() > deadline:
                    success = False
                    break
                    
                # Find the vertex with minimum degree
                min_deg = min(deg)
                
                # If min degree is already sufficient, we can stop adding?
                # No, we want to maximize n, but for a fixed n, we just need min_deg >= needed.
                # However, adding more edges might block future additions?
                # No, adding edges only restricts future additions.
                # So if we reach min_deg >= needed, we can stop and check if it's valid.
                # But we must ensure we don't add an edge that creates a C4.
                # So we stop adding once min_deg >= needed?
                # Wait, if we stop early, we might have a valid graph.
                # But maybe we can add more edges to make it "more" valid? No, validity is binary.
                # So if min_deg >= needed, we are done for this restart.
                
                if min_deg >= min_deg_needed:
                    break
                
                # Identify vertices with min degree
                min_vertices = [i for i in range(target_n) if deg[i] == min_deg]
                
                # Try to find a valid edge incident to one of these vertices
                added = False
                # Shuffle min_vertices to randomize choice
                random.shuffle(min_vertices)
                
                # We will scan all edges to find a valid one incident to min_vertices
                # To speed up, we can pre-filter edges incident to min_vertices
                # But M is small, so just scan all.
                
                # Prioritize edges connecting two low-degree vertices?
                # Let's just find ANY valid edge incident to a min-degree vertex.
                
                candidate_edges = []
                for u, v in all_edges:
                    # Check if edge is already present
                    if (adj[u] >> v) & 1:
                        continue
                    # Check if it involves a min-degree vertex
                    if u not in min_vertices and v not in min_vertices:
                        continue
                    
                    # Check if adding (u,v) creates a C4
                    # Condition: No w in N(u) and x in N(v) such that (w,x) in E
                    # i.e., for all w in N(u), adj[w] & adj[v] == 0
                    # Note: N(u) and N(v) do not include u and v themselves.
                    
                    valid = True
                    # Iterate neighbors of u
                    mask_u = adj[u]
                    while mask_u:
                        lsb = mask_u & -mask_u
                        w = lsb.bit_length() - 1
                        # Check if w is connected to any neighbor of v
                        if adj[w] & adj[v]:
                            valid = False
                            break
                        mask_u ^= lsb
                    
                    if valid:
                        candidate_edges.append((u, v))
                
                if not candidate_edges:
                    # No valid edge incident to min-degree vertices
                    # Can we add an edge not incident to them?
                    # It won't help min degree, but might be necessary?
                    # No, if we can't increase min degree, we are stuck at this min degree.
                    # If min_deg < needed, we fail.
                    success = False
                    break
                
                # Pick a random candidate
                u, v = random.choice(candidate_edges)
                
                # Add edge
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
                added = True
                
                if not added:
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
