import time
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # Precompute comparability matrix
    i_vals = np.arange(total_masks, dtype=np.int32)[:, None]
    j_vals = np.arange(total_masks, dtype=np.int32)[None, :]
    
    cond1 = (i_vals & j_vals) == i_vals
    cond2 = (i_vals & j_vals) == j_vals
    comp = cond1 | cond2
    np.fill_diagonal(comp, False)
    
    def calc_deg(fam):
        fam_arr = np.array(fam, dtype=np.int32)
        deg = comp[:, fam_arr].sum(axis=1).astype(np.int32)
        return deg

    def get_score(fam, deg):
        return int(np.sum(deg) / 2)

    def greedy_init():
        fam = []
        in_fam_mask = np.zeros(total_masks, dtype=bool)
        gain = np.zeros(total_masks, dtype=np.int32)
        
        for _ in range(m):
            gain_masked = gain.copy()
            gain_masked[in_fam_mask] = -1
            v = int(np.argmax(gain_masked))
            
            fam.append(v)
            in_fam_mask[v] = True
            
            gain += comp[:, v]
            
        return fam

    def local_search(fam):
        fam = list(fam)
        deg = calc_deg(fam)
        score = get_score(fam, deg)
        fam_set = set(fam)
        
        in_fam_mask = np.zeros(total_masks, dtype=bool)
        for f in fam:
            in_fam_mask[f] = True
        not_in_fam_mask = ~in_fam_mask
        
        T = 10.0
        cooling_rate = 0.95
        max_no_improve = 100
        no_improve_count = 0
        best_score_so_far = score
        best_fam_so_far = list(fam)
        
        fam_arr = np.array(fam, dtype=np.int32)
        
        while time.time() < deadline - 3:
            improved = False
            best_delta = 0
            best_old = None
            best_new = None
            
            deg_candidates = deg.copy()
            deg_candidates[in_fam_mask] = -1
            
            K = 200
            if K > total_masks: K = total_masks
            top_indices = np.argpartition(deg_candidates, -K)[-K:]
            
            # Precompute degrees of current family members
            deg_fam = deg[fam_arr]
            
            for m_new in top_indices:
                if not_in_fam_mask[m_new]:
                    # comp[m_new, fam_arr] is a vector of 0/1
                    comp_m_new_fam = comp[m_new, fam_arr]
                    costs = deg_fam + comp_m_new_fam
                    min_cost_idx = np.argmin(costs)
                    min_cost = costs[min_cost_idx]
                    delta = deg[m_new] - min_cost
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_old = fam_arr[min_cost_idx]
                        best_new = m_new
            
            if best_delta > 0:
                # Update deg
                deg -= comp[:, best_old]
                deg += comp[:, best_new]
                
                # Update fam
                idx = list(fam).index(best_old)
                fam[idx] = best_new
                fam_arr[idx] = best_new
                fam_set.remove(best_old)
                fam_set.add(best_new)
                
                in_fam_mask[best_old] = False
                in_fam_mask[best_new] = True
                not_in_fam_mask[best_old] = True
                not_in_fam_mask[best_new] = False
                
                score += best_delta
                improved = True
                no_improve_count = 0
                
                if score > best_score_so_far:
                    best_score_so_far = score
                    best_fam_so_far = list(fam)
                    
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                
                # Random perturbation / 2-swap attempt?
                # Let's try a random 1-swap if T is high
                if T > 1.0:
                    # Pick random old
                    idx_old = np.random.choice(m)
                    m_old = fam[idx_old]
                    
                    # Pick random new from top candidates or random?
                    # Let's pick from top candidates
                    if len(top_indices) > 0:
                        m_new = top_indices[np.random.choice(len(top_indices))]
                        if not_in_fam_mask[m_new]:
                             # Calculate delta
                             comp_m_new_fam = comp[m_new, fam_arr]
                             costs = deg_fam + comp_m_new_fam
                             # We are swapping m_old specifically
                             # delta = deg[m_new] - (deg[m_old] + comp[m_new, m_old])
                             c = 1 if comp[m_new, m_old] else 0
                             delta = deg[m_new] - deg[m_old] - c
                             
                             if delta >= 0 or np.random.random() < np.exp(delta / T):
                                 deg -= comp[:, m_old]
                                 deg += comp[:, m_new]
                                 
                                 fam[idx_old] = m_new
                                 fam_arr[idx_old] = m_new
                                 fam_set.remove(m_old)
                                 fam_set.add(m_new)
                                 
                                 in_fam_mask[m_old] = False
                                 in_fam_mask[m_new] = True
                                 not_in_fam_mask[m_old] = True
                                 not_in_fam_mask[m_new] = False
                                 
                                 score += delta
                                 if score > best_score_so_far:
                                     best_score_so_far = score
                                     best_fam_so_far = list(fam)
                                 
                T *= cooling_rate
            
        return best_fam_so_far, best_score_so_far

    best_score = 0
    best_fam = None
    
    rng = np.random.default_rng(42)
    
    # Run greedy init once
    fam = greedy_init()
    cur_fam, cur_score = local_search(fam)
    
    if cur_score > best_score:
        best_score = cur_score
        best_fam = cur_fam
        
    # Try a few random restarts if time permits
    while time.time() < deadline - 2:
        fam = rng.choice(total_masks, size=m, replace=False).tolist()
        cur_fam, cur_score = local_search(fam)
        
        if cur_score > best_score:
            best_score = cur_score
            best_fam = cur_fam
            
        if best_score > 700:
             break

    result = []
    for mask in best_fam:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
