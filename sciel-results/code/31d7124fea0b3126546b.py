import time
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Precompute comparability matrix
    # i, j are masks. comparable if (i & j) == i or (i & j) == j
    i_vals = np.arange(total_masks, dtype=np.int32)[:, None]
    j_vals = np.arange(total_masks, dtype=np.int32)[None, :]
    
    # Vectorized comparability check
    # (i & j) == i  OR  (i & j) == j
    cond1 = (i_vals & j_vals) == i_vals
    cond2 = (i_vals & j_vals) == j_vals
    comp = cond1 | cond2
    np.fill_diagonal(comp, False)
    
    # Helper to calculate initial degree array for a family
    def calc_deg(fam):
        # fam is a list of indices
        # deg[m] = sum(comp[m, f] for f in fam)
        # Using matrix multiplication or sum
        # comp is (4096, 4096)
        # fam is list of cols
        # deg = comp[:, fam].sum(axis=1)
        fam_arr = np.array(fam, dtype=np.int32)
        deg = comp[:, fam_arr].sum(axis=1).astype(np.int32)
        return deg

    # Helper to get score
    def get_score(fam, deg):
        # Sum of degrees / 2? 
        # deg[m] counts edges to other members.
        # Sum(deg) = 2 * edges
        return int(np.sum(deg) / 2)

    # Local Search
    def local_search(fam):
        fam = list(fam)
        deg = calc_deg(fam)
        score = get_score(fam, deg)
        fam_set = set(fam)
        
        # Precompute list of all masks not in fam for faster iteration?
        # Or use boolean mask
        in_fam_mask = np.zeros(total_masks, dtype=bool)
        for f in fam:
            in_fam_mask[f] = True
        not_in_fam_mask = ~in_fam_mask
        
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_old = None
            best_new = None
            
            # Get top K candidates for new mask based on degree
            # We want high deg[m_new]
            # Mask out those in fam
            deg_candidates = deg.copy()
            deg_candidates[in_fam_mask] = -1
            
            # Get top 100 indices
            K = 100
            if K > total_masks: K = total_masks
            top_indices = np.argpartition(deg_candidates, -K)[-K:]
            
            # Check these candidates against all old masks
            for m_new in top_indices:
                if not_in_fam_mask[m_new]:
                    # For each m_old in fam
                    # delta = deg[m_new] - comp[m_new, m_old] - deg[m_old]
                    # We want max delta
                    # deg[m_new] is constant for this m_new
                    # So we want min (comp[m_new, m_old] + deg[m_old])
                    # Or just iterate
                    for m_old in fam:
                        c = 1 if comp[m_new, m_old] else 0
                        delta = deg[m_new] - c - deg[m_old]
                        if delta > best_delta:
                            best_delta = delta
                            best_old = m_old
                            best_new = m_new
            
            if best_delta > 0:
                # Perform swap
                # Update deg
                # Remove best_old, add best_new
                # deg changes by -comp[:, best_old] + comp[:, best_new]
                deg -= comp[:, best_old]
                deg += comp[:, best_new]
                
                # Update fam
                fam[fam.index(best_old)] = best_new
                fam_set.remove(best_old)
                fam_set.add(best_new)
                
                # Update masks
                in_fam_mask[best_old] = False
                in_fam_mask[best_new] = True
                not_in_fam_mask[best_old] = True
                not_in_fam_mask[best_new] = False
                
                score += best_delta
                improved = True
                
                if time.time() > deadline - 5:
                    break
        
        return fam, score

    best_score = 0
    best_fam = None
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    
    # Initial heuristic: Start with a random set, or maybe a structured one?
    # Random is fine with good local search.
    
    while time.time() < deadline - 2:
        # Generate random family
        fam = rng.choice(total_masks, size=m, replace=False).tolist()
        
        cur_fam, cur_score = local_search(fam)
        
        if cur_score > best_score:
            best_score = cur_score
            best_fam = cur_fam
            
        # If we found a very high score, we might stop early or perturb
        # Max possible score? 40 * 39 / 2 = 780.
        # If we reach near 780, stop.
        if best_score > 750:
             break

    # Convert to lists of vertices
    result = []
    for mask in best_fam:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
