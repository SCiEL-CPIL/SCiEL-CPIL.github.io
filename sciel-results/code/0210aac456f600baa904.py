import time
import random
import math
import sys

_DEADLINE = 145.0

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    rng = random.Random(42)
    
    # Initialize adj
    adj = [[0] * n for _ in range(n)]
    # K_8,8
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
    # Add one internal edge to make 65 edges (K_8,8 has 64)
    adj[0][1] = 1
    adj[1][0] = 1
    
    # Calculate degrees
    deg = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                deg[i] += 1
                
    # Identify all edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append((i, j))
                
    # Identify all non-edges
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                non_edges.append((i, j))
                
    # Function to build triangle data from scratch
    def build_tri_data(adj, deg):
        triangles = set()
        # For each edge (i,j), find common neighbors k > j
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    # Common neighbors
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            triangles.add((i, j, k))
        # Calculate sums
        tri_sums = {}
        count_sum = [0] * 50 # Max sum is 3*15=45, so 50 is safe
        max_sum = 0
        for t in triangles:
            s = deg[t[0]] + deg[t[1]] + deg[t[2]]
            tri_sums[t] = s
            count_sum[s] += 1
            if s > max_sum:
                max_sum = s
        return triangles, tri_sums, count_sum, max_sum

    triangles, tri_sums, count_sum, max_sum = build_tri_data(adj, deg)
    
    best_max_sum = max_sum
    best_adj = [row[:] for row in adj]
    
    # Simulated Annealing
    T_start = 5.0
    T_end = 0.01
    total_steps = 2000000
    steps = 0
    
    while steps < total_steps:
        if time.time() > _DEADLINE:
            break
            
        # Pick a random present edge and a random absent edge
        e1 = rng.choice(edges)
        e2 = rng.choice(non_edges)
        
        u1, v1 = e1
        u2, v2 = e2
        
        # Save state for potential revert
        old_deg_u1 = deg[u1]
        old_deg_v1 = deg[v1]
        old_deg_u2 = deg[u2]
        old_deg_v2 = deg[v2]
        
        # 1. Identify triangles to remove (containing u1, v1)
        removed_tris = []
        for k in range(n):
            if k == u1 or k == v1: continue
            if adj[u1][k] and adj[v1][k]:
                t = tuple(sorted((u1, v1, k)))
                if t in triangles:
                    removed_tris.append((t, tri_sums[t]))
        
        # 2. Identify triangles to add (containing u2, v2)
        # Note: u2, v2 are not connected yet in adj, but we are about to connect them.
        # Common neighbors of u2 and v2 in the CURRENT graph (before adding edge)
        added_tris = []
        for k in range(n):
            if k == u2 or k == v2: continue
            if adj[u2][k] and adj[v2][k]:
                t = tuple(sorted((u2, v2, k)))
                # Calculate sum with NEW degrees
                # deg[u2] and deg[v2] will be incremented by 1
                # Other vertex k degree remains same
                new_s = (deg[u2] + 1) + (deg[v2] + 1) + deg[k]
                added_tris.append((t, new_s))
        
        # 3. Identify triangles to modify
        # These contain u1, v1, u2, or v2
        # But do NOT contain edge (u1, v1) [already removed]
        # And do NOT contain edge (u2, v2) [already added]
        # We need to update their sums because deg[u1], deg[v1], deg[u2], deg[v2] change.
        
        # First, update degrees to NEW values for calculation
        new_deg_u1 = old_deg_u1 - 1
        new_deg_v1 = old_deg_v1 - 1
        new_deg_u2 = old_deg_u2 + 1
        new_deg_v2 = old_deg_v2 + 1
        
        modified_tris = []
        for t in triangles:
            # Check if t is affected
            # t is a tuple of 3 vertices
            # If t contains u1 or v1, and is not in removed_tris
            # If t contains u2 or v2, and is not in added_tris
            
            # Efficient check:
            # t is in removed_tris if u1 in t and v1 in t
            # t is in added_tris if u2 in t and v2 in t
            
            if (u1 in t or v1 in t or u2 in t or v2 in t):
                # Check if it's a removed triangle
                if u1 in t and v1 in t:
                    continue
                # Check if it's an added triangle (shouldn't be in `triangles` yet, but safe check)
                if u2 in t and v2 in t:
                    continue
                
                old_s = tri_sums[t]
                # Calculate new sum
                new_s = 0
                for v in t:
                    if v == u1:
                        new_s += new_deg_u1
                    elif v == v1:
                        new_s += new_deg_v1
                    elif v == u2:
                        new_s += new_deg_u2
                    elif v == v2:
                        new_s += new_deg_v2
                    else:
                        new_s += deg[v]
                
                if new_s != old_s:
                    modified_tris.append((t, old_s, new_s))
        
        # Apply changes to data structures
        # 1. Handle removed
        for t, old_s in removed_tris:
            triangles.discard(t)
            del tri_sums[t]
            count_sum[old_s] -= 1
            
        # 2. Handle added
        for t, new_s in added_tris:
            triangles.add(t)
            tri_sums[t] = new_s
            count_sum[new_s] += 1
            
        # 3. Handle modified
        for t, old_s, new_s in modified_tris:
            count_sum[old_s] -= 1
            count_sum[new_s] += 1
            tri_sums[t] = new_s
            
        # Update degrees
        deg[u1] = new_deg_u1
        deg[v1] = new_deg_v1
        deg[u2] = new_deg_u2
        deg[v2] = new_deg_v2
        
        # Update adj
        adj[u1][v1] = 0
        adj[v1][u1] = 0
        adj[u2][v2] = 1
        adj[v2][u2] = 1
        
        # Update edge lists
        # Swap e1 and e2 in lists
        # Find index of e1 in edges
        idx1 = edges.index(e1)
        edges[idx1] = e2
        # Find index of e2 in non_edges
        idx2 = non_edges.index(e2)
        non_edges[idx2] = e1
        
        # Calculate new_max_sum
        # If we removed the max, we might need to scan.
        # If we added a higher max, update.
        # Simplest: scan count_sum from high to low.
        new_max_sum = 0
        for s in range(len(count_sum)-1, -1, -1):
            if count_sum[s] > 0:
                new_max_sum = s
                break
        
        # Acceptance
        accepted = False
        if new_max_sum < best_max_sum:
            best_max_sum = new_max_sum
            best_adj = [row[:] for row in adj]
            accepted = True
        else:
            delta = new_max_sum - best_max_sum
            if delta == 0:
                accepted = True # Accept neutral moves to explore
            else:
                progress = steps / total_steps
                T = T_start * math.exp(-10.0 * progress)
                if T < T_end: T = T_end
                if rng.random() < math.exp(-delta / T):
                    accepted = True
        
        if not accepted:
            # Revert
            # 1. Revert adj
            adj[u1][v1] = 1
            adj[v1][u1] = 1
            adj[u2][v2] = 0
            adj[v2][u2] = 0
            
            # 2. Revert degrees
            deg[u1] = old_deg_u1
            deg[v1] = old_deg_v1
            deg[u2] = old_deg_u2
            deg[v2] = old_deg_v2
            
            # 3. Revert edge lists
            edges[idx1] = e1
            non_edges[idx2] = e2
            
            # 4. Revert triangle data
            # Re-add removed
            for t, old_s in removed_tris:
                triangles.add(t)
                tri_sums[t] = old_s
                count_sum[old_s] += 1
                
            # Remove added
            for t, new_s in added_tris:
                triangles.discard(t)
                del tri_sums[t]
                count_sum[new_s] -= 1
                
            # Revert modified
            for t, old_s, new_s in modified_tris:
                count_sum[new_s] -= 1
                count_sum[old_s] += 1
                tri_sums[t] = old_s
                
        steps += 1
        
    # Final Polish
    # Local search: accept only if strictly better
    # Use the best_adj found so far
    # Rebuild state from best_adj
    adj = [row[:] for row in best_adj]
    deg = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                deg[i] += 1
                
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append((i, j))
                
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                non_edges.append((i, j))
                
    triangles, tri_sums, count_sum, max_sum = build_tri_data(adj, deg)
    current_max = max_sum
    
    polish_steps = 0
    max_polish_steps = 50000
    no_improve_count = 0
    threshold = 5000
    
    while polish_steps < max_polish_steps and time.time() < _DEADLINE:
        if no_improve_count > threshold:
            break
            
        e1 = rng.choice(edges)
        e2 = rng.choice(non_edges)
        
        u1, v1 = e1
        u2, v2 = e2
        
        # Save state
        old_deg_u1 = deg[u1]
        old_deg_v1 = deg[v1]
        old_deg_u2 = deg[u2]
        old_deg_v2 = deg[v2]
        
        # Identify removed
        removed_tris = []
        for k in range(n):
            if k == u1 or k == v1: continue
            if adj[u1][k] and adj[v1][k]:
                t = tuple(sorted((u1, v1, k)))
                if t in triangles:
                    removed_tris.append((t, tri_sums[t]))
                    
        # Identify added
        added_tris = []
        new_deg_u2 = old_deg_u2 + 1
        new_deg_v2 = old_deg_v2 + 1
        for k in range(n):
            if k == u2 or k == v2: continue
            if adj[u2][k] and adj[v2][k]:
                t = tuple(sorted((u2, v2, k)))
                new_s = new_deg_u2 + new_deg_v2 + deg[k]
                added_tris.append((t, new_s))
                
        # Identify modified
        new_deg_u1 = old_deg_u1 - 1
        new_deg_v1 = old_deg_v1 - 1
        
        modified_tris = []
        for t in triangles:
            if (u1 in t or v1 in t or u2 in t or v2 in t):
                if u1 in t and v1 in t:
                    continue
                if u2 in t and v2 in t:
                    continue
                
                old_s = tri_sums[t]
                new_s = 0
                for v in t:
                    if v == u1: new_s += new_deg_u1
                    elif v == v1: new_s += new_deg_v1
                    elif v == u2: new_s += new_deg_u2
                    elif v == v2: new_s += new_deg_v2
                    else: new_s += deg[v]
                
                if new_s != old_s:
                    modified_tris.append((t, old_s, new_s))
                    
        # Apply
        for t, old_s in removed_tris:
            triangles.discard(t)
            del tri_sums[t]
            count_sum[old_s] -= 1
        for t, new_s in added_tris:
            triangles.add(t)
            tri_sums[t] = new_s
            count_sum[new_s] += 1
        for t, old_s, new_s in modified_tris:
            count_sum[old_s] -= 1
            count_sum[new_s] += 1
            tri_sums[t] = new_s
            
        deg[u1] = new_deg_u1
        deg[v1] = new_deg_v1
        deg[u2] = new_deg_u2
        deg[v2] = new_deg_v2
        
        adj[u1][v1] = 0
        adj[v1][u1] = 0
        adj[u2][v2] = 1
        adj[v2][u2] = 1
        
        idx1 = edges.index(e1)
        edges[idx1] = e2
        idx2 = non_edges.index(e2)
        non_edges[idx2] = e1
        
        new_max_sum = 0
        for s in range(len(count_sum)-1, -1, -1):
            if count_sum[s] > 0:
                new_max_sum = s
                break
                
        if new_max_sum < current_max:
            current_max = new_max_sum
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Revert
            adj[u1][v1] = 1
            adj[v1][u1] = 1
            adj[u2][v2] = 0
            adj[v2][u2] = 0
            
            deg[u1] = old_deg_u1
            deg[v1] = old_deg_v1
            deg[u2] = old_deg_u2
            deg[v2] = old_deg_v2
            
            edges[idx1] = e1
            non_edges[idx2] = e2
            
            for t, old_s in removed_tris:
                triangles.add(t)
                tri_sums[t] = old_s
                count_sum[old_s] += 1
            for t, new_s in added_tris:
                triangles.discard(t)
                del tri_sums[t]
                count_sum[new_s] -= 1
            for t, old_s, new_s in modified_tris:
                count_sum[new_s] -= 1
                count_sum[old_s] += 1
                tri_sums[t] = old_s
                
        polish_steps += 1
        
    edges_out = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges_out.append([i, j])
                
    return {'n': n, 'edges': edges_out}
