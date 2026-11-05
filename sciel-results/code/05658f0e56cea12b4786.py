import numpy as np
import time

def priority(adjacency):
    """
    Returns a priority score for each vertex.
    Higher score means the vertex should be picked earlier by the greedy decoder.
    Strategy: Minimize Core Number, then Minimize Degree.
    """
    start_time = time.time()
    n = adjacency.shape[0]
    
    # 1. Compute Degrees
    deg = adjacency.sum(axis=1).astype(int)
    
    # 2. Compute Core Numbers (K-Core Decomposition)
    # We use a bucket-based approach for efficiency.
    core = np.zeros(n, dtype=int)
    
    # Bucket vertices by their current degree
    # Max degree is n-1
    buckets = [[] for _ in range(n)]
    for i in range(n):
        buckets[deg[i]].append(i)
        
    max_k = n - 1
    
    # Peeling process
    # We process from highest degree down
    # Actually, standard k-core peeling starts from low degree and removes, 
    # but to assign core numbers, we can do it by finding the max k such that 
    # the subgraph induced by vertices with degree >= k contains the vertex.
    
    # Efficient K-Core Algorithm:
    # 1. Initialize deg[i]
    # 2. Put i in bucket[deg[i]]
    # 3. For k from n-1 down to 1:
    #    While bucket[k] is not empty:
    #       v = bucket[k].pop()
    #       if core[v] == 0:
    #           core[v] = k
    #           for each neighbor u of v:
    #               if core[u] == 0:
    #                   deg[u] -= 1
    #                   if deg[u] < k: # It dropped below current k, it will be handled in lower k iterations?
    #                       # Actually, we just update deg and move to new bucket?
    #                       # Standard algorithm:
    #                       # When we remove v from k-core, we decrement neighbors.
    #                       # If neighbor's degree drops, it might fall out of k-core.
    #                       # But we are iterating k downwards.
    #                       # If deg[u] becomes < k, it will be assigned a core < k.
    #                       # We should move it to bucket[deg[u]]?
    #                       # Yes.
    #                       buckets[deg[u]].append(u)
    
    # Let's refine the K-Core loop.
    # The standard algorithm processes vertices in order of increasing degree for removal.
    # But for assignment, we can do:
    # 1. deg[i] = initial degree
    # 2. buckets[deg[i]].append(i)
    # 3. for k in range(n-1, 0, -1):
    #      while bucket[k] is not empty:
    #          v = bucket[k].pop()
    #          if core[v] != 0: continue # already assigned
    #          # Check if v still has degree >= k?
    #          # Actually, if it's in bucket[k], its current degree is k.
    #          # But neighbors might have been removed, lowering its degree?
    #          # No, if neighbor was removed, it was from a higher k?
    #          # If neighbor was removed from k' > k, then v's degree dropped.
    #          # If v's degree dropped, it should have been moved to a lower bucket.
    #          # So if v is in bucket[k], its current degree is exactly k.
    #          core[v] = k
    #          for u in neighbors(v):
    #              if core[u] == 0:
    #                  deg[u] -= 1
    #                  buckets[deg[u]].append(u)
    
    # This works.
    
    neighbors = [np.where(adjacency[i, :])[0] for i in range(n)]
    
    for k in range(n - 1, 0, -1):
        while buckets[k]:
            v = buckets[k].pop()
            if core[v] != 0:
                continue
            # Verify degree is still k? 
            # If deg[v] < k, it was moved. If deg[v] > k, it shouldn't be here.
            # But wait, if we decrement deg[u], we add to buckets[deg[u]].
            # So buckets[k] contains vertices with current degree k.
            core[v] = k
            for u in neighbors[v]:
                if core[u] == 0:
                    deg[u] -= 1
                    # deg[u] is now the new degree
                    buckets[deg[u]].append(u)
    
    # 3. Construct Score
    # We want High Score for Low Core, then Low Degree.
    # Score = -(Core * LargeConstant + OriginalDegree)
    # OriginalDegree is stored in 'adjacency.sum(axis=1)' but we modified 'deg'.
    # Let's recompute original degree or save it.
    orig_deg = adjacency.sum(axis=1).astype(int)
    
    # Large constant to ensure Core dominates
    # Max degree is n-1 (199).
    # So constant 1000 is safe.
    penalty = core * 1000 + orig_deg
    score = -penalty.astype(float)
    
    return score
