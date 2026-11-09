import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 0
    best_edges = []
    
    # Try n=47 first, then 46
    for target_n in [47, 46]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        # Use Simulated Annealing
        # We need a graph on target_n vertices with min_deg >= min_deg_needed and C4-free.
        
        # Initialize with a random graph
        # p should be high enough to get min_deg, but low enough to avoid C4s.
        # For n=46, min_deg=7. p ~ 7/45 ~ 0.15.
        # C4-free graphs are sparse.
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            random.seed(restart * 100 + target_n)
            
            # Initialize A
            # Start with a sparse random graph
            p = min_deg_needed / (target_n - 1) * 1.2
            A = np.zeros((target_n, target_n), dtype=bool)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < p:
                        A[i, j] = True
                        A[j, i] = True
            
            # Calculate degrees
            deg = A.sum(axis=1)
            
            # Ensure min degree is at least min_deg_needed
            # If not, add random edges (ignoring C4 for now, will fix later)
            for _ in range(50):
                if time.time() > deadline:
                    break
                low = [i for i in range(target_n) if deg[i] < min_deg_needed]
                if not low:
                    break
                u = random.choice(low)
                # Find non-neighbor
                non_neigh = np.where(~A[u] & (np.arange(target_n) != u))[0]
                if len(non_neigh) == 0:
                    continue
                v = random.choice(non_neigh)
                A[u, v] = True
                A[v, u] = True
                deg[u] += 1
                deg[v] += 1
            
            # Simulated Annealing to remove C4s while maintaining degree
            # Energy: sum of (C_uv - 1)^2 for C_uv > 1
            # C = A @ A
            # We want C_uv <= 1 for all u != v.
            
            temp = 10.0
            cooling = 0.95
            max_iters = 2000
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if temp < 0.01:
                    break
                
                # Pick random edge to flip
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                # Calculate change in C4 count
                # This is expensive to do fully.
                # Approximation: Check if flipping (u,v) reduces C4s.
                # Flipping (u,v) changes A.
                # Let's just do a local check.
                
                # Current C4 count for edges involving u or v?
                # Too slow.
                
                # Alternative: Just try to remove edges that are in many C4s.
                # Compute C = A @ A
                C = A @ A
                # C_uv is number of common neighbors.
                # Violations: C_uv >= 2.
                
                # Find edge (u,v) to remove.
                # An edge (u,v) is "bad" if it contributes to C4s.
                # Removing (u,v) reduces C_uw and C_vw for all w in N(u) U N(v).
                
                # Let's pick a random edge that is part of a C4.
                # Find a pair (x,y) with C_xy >= 2.
                # Then any edge (x,z) or (y,z) with z in common neighbors is part of a C4.
                
                # Efficiently:
                # Find all pairs (x,y) with C_xy >= 2.
                bad_pairs = np.where(C >= 2)
                if len(bad_pairs[0]) == 0:
                    # Check degree
                    if np.min(deg) >= min_deg_needed:
                        found = True
                        break
                    else:
                        # Need to add edges, but we are C4 free?
                        # If C4 free and degree low, we are stuck or need to restructure.
                        # For now, break SA and try next restart.
                        break
                
                # Pick a random bad pair
                idx = random.randint(0, len(bad_pairs[0]) - 1)
                x = bad_pairs[0][idx]
                y = bad_pairs[1][idx]
                
                # Common neighbors of x and y
                common = np.where(A[x] & A[y])[0]
                # Pick a common neighbor z
                z = random.choice(common)
                
                # The cycle is x-z-y-w-x.
                # We can remove edge (x,z) or (z,y) or (y,w) or (w,x).
                # Let's remove (x,z).
                
                # Check if removing (x,z) violates degree
                if deg[x] > min_deg_needed and deg[z] > min_deg_needed:
                    A[x, z] = False
                    A[z, x] = False
                    deg[x] -= 1
                    deg[z] -= 1
                else:
                    # Try removing (z,y)
                    if deg[z] > min_deg_needed and deg[y] > min_deg_needed:
                        A[z, y] = False
                        A[y, z] = False
                        deg[z] -= 1
                        deg[y] -= 1
                    else:
                        # Try removing (y,w) where w is another common neighbor?
                        # Or just accept failure for this restart
                        pass
                
                temp *= cooling
            
            # Final Check
            if found:
                # Verify
                C = A @ A
                is_c4_free = True
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if C[i, j] >= 2:
                            is_c4_free = False
                            break
                    if not is_c4_free:
                        break
                
                if is_c4_free and np.min(deg) >= min_deg_needed:
                    final_edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if A[i, j]:
                                final_edges.append([i, j])
                    
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = final_edges
                    found = True
                    break # Break restart loop
        
        if best_n == 47:
            break
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback
        return {'n': 39, 'edges': []}
