import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    # 1. Norm 2 solution (110 points)
    # Vectors are +/- e_i +/- e_j for i < j
    pts2 = []
    for i in range(d):
        for j in range(i+1, d):
            # 4 combinations of signs
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    pts2.append(v)
    pts2 = np.array(pts2)
    best_k = len(pts2)
    best_pts = pts2
    
    # 2. Try Norm 3
    # Vectors are +/- e_i +/- e_j +/- e_k for i < j < k
    pts3 = []
    idxs3 = []
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            pts3.append(v)
                            idxs3.append((i,j,k,s1,s2,s3))
    pts3 = np.array(pts3)
    n3 = len(pts3)
    
    # Build conflict graph
    # Conflict if dot product == 2
    # We can compute dot products efficiently
    # But 1320x1320 is ~1.7M dots. Fast enough.
    # Or use the property: conflict if share 2 coords with same sign.
    
    # Greedy MIS
    start_time = time.time()
    deadline = start_time + 30 # 30 seconds for search
    
    best_mis_size = 0
    best_mis_indices = []
    
    for restart in range(1000):
        if time.time() > deadline:
            break
        random.shuffle(idxs3)
        available = set(range(n3))
        chosen = []
        
        # Pre-compute neighbors? 
        # For 1320 nodes, checking all pairs in greedy step is O(N^2) per step.
        # Better to precompute adjacency list once.
        
        # Let's precompute adjacency list for all norm 3 vectors
        # Do this outside the restart loop if not done.
        
        # Actually, let's just do it once before the loop.
        
    # Precompute adjacency
    adj = [set() for _ in range(n3)]
    # Vectorized dot product check?
    # pts3 is (1320, 11)
    # Dots = pts3 @ pts3.T
    # Conflicts where Dots == 2
    
    Dots = pts3 @ pts3.T
    # Dots is (1320, 1320)
    # Mask where Dots == 2
    # Note: Dots[i,i] = 3. We don't care about self.
    
    for i in range(n3):
        # Find j such that Dots[i,j] == 2
        # This is O(N) per i, total O(N^2)
        # 1.7M ops is fine.
        neighbors = np.where(Dots[i] == 2)[0]
        for j in neighbors:
            if j != i:
                adj[i].add(j)
                
    # Now run greedy with restarts
    for restart in range(5000):
        if time.time() > deadline:
            break
        order = np.random.permutation(n3)
        available = set(range(n3))
        chosen = []
        
        for idx in order:
            if idx not in available:
                continue
            # Check if idx conflicts with any chosen
            # Actually, standard greedy:
            # If idx is available, add it, and remove it and its neighbors from available.
            # This works if we process in random order.
            chosen.append(idx)
            available.remove(idx)
            for neighbor in adj[idx]:
                if neighbor in available:
                    available.remove(neighbor)
            
        if len(chosen) > best_mis_size:
            best_mis_size = len(chosen)
            best_mis_indices = chosen
            
        if best_mis_size > best_k:
            # Early exit if we already beat 110 significantly?
            # No, keep searching for a bit more.
            pass

    if best_mis_size > best_k:
        best_pts = pts3[best_mis_indices]
        best_k = best_mis_size

    # Final verification
    # ...
    
    return best_pts
