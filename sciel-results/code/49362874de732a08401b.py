import time
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute comparability matrix
    # i, j are masks. comparable if (i & j) == i or (i & j) == j
    i_vals = np.arange(total_masks, dtype=np.int32)[:, None]
    j_vals = np.arange(total_masks, dtype=np.int32)[None, :]
    
    # Vectorized comparability check
    cond1 = (i_vals & j_vals) == i_vals
    cond2 = (i_vals & j_vals) == j_vals
    comp = (cond1 | cond2).astype(np.uint8)
    np.fill_diagonal(comp, 0)
    
    # Local Search
    def local_search(fam):
        fam = list(fam)
        fam_set = set(fam)
        in_fam_mask = np.zeros(total_masks, dtype=np.uint8)
        for f in fam:
            in_fam_mask[f] = 1
        
        # deg[m] = sum(comp[m, f] for f in fam)
        # deg = comp.dot(in_fam_mask)
        deg = comp.dot(in_fam_mask)
        score = int(np.sum(deg) // 2)
        
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_old = None
            best_new = None
            
            # Try removing each u in fam
            for u in fam:
                # M_u is in_fam_mask with u removed
                M_u = in_fam_mask.copy()
                M_u[u] = 0
                
                # scores_v[v] = sum(comp[v, w] for w in M_u)
                scores_v = comp.dot(M_u)
                
                # Mask out v in fam
                scores_v[in_fam_mask > 0] = 0
                
                # Find best v
                v = np.argmax(scores_v)
                if scores_v[v] > 0:
                    # delta = (scores_v[v] - deg[u]) / 2
                    # deg[u] is sum(comp[u, w] for w in fam)
                    # Note: deg[u] includes comp[u, u]=0, so it is sum over fam \ {u}
                    # So delta = (scores_v[v] - deg[u]) / 2
                    delta = (int(scores_v[v]) - int(deg[u])) // 2
                    if delta > best_delta:
                        best_delta = delta
                        best_old = u
                        best_new = v
            
            if best_delta > 0:
                # Perform swap
                u = best_old
                v = best_new
                
                # Update deg
                # deg changes by -comp[:, u] + comp[:, v]
                deg -= comp[:, u]
                deg += comp[:, v]
                
                # Update fam
                fam[fam.index(u)] = v
                fam_set.remove(u)
                fam_set.add(v)
                
                # Update masks
                in_fam_mask[u] = 0
                in_fam_mask[v] = 1
                
                score += best_delta
                improved = True
                
                if time.time() > deadline - 5:
                    break
        
        return fam, score

    best_score = 0
    best_fam = None
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 2:
        # Generate random family
        fam = rng.choice(total_masks, size=m, replace=False).tolist()
        
        cur_fam, cur_score = local_search(fam)
        
        if cur_score > best_score:
            best_score = cur_score
            best_fam = cur_fam
            
        if best_score > 750:
            break

    # Convert to lists of vertices
    result = []
    for mask in best_fam:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
