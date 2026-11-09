import numpy as np
import time
import itertools

def construct():
    deadline = time.time() + 140
    n = 16
    edge_idx = {}
    edges = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            edges.append((i,j))
            idx += 1
    
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a,b,c,d = quad
        e_idx = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                  edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(e_idx)
    k4s = np.array(k4s, dtype=np.int32)
    num_k4s = len(k4s)
    
    edge_to_k4s = [[] for _ in range(120)]
    for k, e_idx in enumerate(k4s):
        for e in e_idx:
            edge_to_k4s[e].append(k)
    edge_to_k4s = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]
    
    def get_distinct(col_arr, k4_indices):
        if len(k4_indices) == 0:
            return np.array([], dtype=np.int32)
        subset = col_arr[k4s[k4_indices]]
        s = np.sort(subset, axis=1)
        d = np.ones((len(k4_indices),), dtype=np.int32)
        for k in range(1, 6):
            d += (s[:, k] != s[:, k-1]).astype(np.int32)
        return d
    
    best_colors = None
    best_num_colors = 18
    
    rng = np.random.default_rng(42)
    
    for target_k in [15, 14, 13]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple restarts
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            
            if restart == 0:
                cur = np.array([(i+j) % 17 for i,j in edges], dtype=np.int32)
            else:
                cur = rng.integers(0, target_k, size=120).astype(np.int32)
            
            distinct = get_distinct(cur, np.arange(num_k4s))
            bad = np.maximum(0, 5 - distinct)
            cur_bad = int(np.sum(bad ** 2))
            
            cc = np.zeros(target_k, dtype=np.int32)
            for c in cur:
                cc[c] += 1
            nc = int(np.sum(cc > 0))
            
            T = 50.0
            T_min = 0.1
            cooling = 0.9995
            no_improve = 0
            max_no_improve = 30000
            
            while time.time() < deadline - 20 and T > T_min and no_improve < max_no_improve:
                e = int(rng.integers(0, 120))
                old_c = int(cur[e])
                new_c = int(rng.integers(0, target_k))
                if new_c == old_c:
                    continue
                
                affected = edge_to_k4s[e]
                old_distinct_affected = distinct[affected].copy()
                
                cur[e] = new_c
                new_distinct_affected = get_distinct(cur, affected)
                
                old_bad_affected = int(np.sum(np.maximum(0, 5 - old_distinct_affected)**2))
                new_bad_affected = int(np.sum(np.maximum(0, 5 - new_distinct_affected)**2))
                delta_bad = new_bad_affected - old_bad_affected
                
                cc[old_c] -= 1
                cc[new_c] += 1
                old_nc = nc
                new_nc = nc
                if cc[old_c] == 0:
                    new_nc -= 1
                if cc[new_c] == 1:
                    new_nc += 1
                
                delta_color = (max(0, new_nc - target_k) - max(0, old_nc - target_k)) * 1000
                delta = delta_bad + delta_color
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    distinct[affected] = new_distinct_affected
                    cur_bad += delta_bad
                    nc = new_nc
                    no_improve = 0
                    if cur_bad == 0 and nc <= target_k:
                        if nc < best_num_colors:
                            best_num_colors = nc
                            best_colors = cur.copy()
                        break
                else:
                    cur[e] = old_c
                    cc[old_c] += 1
                    cc[new_c] -= 1
                    no_improve += 1
                T *= cooling
            
            if cur_bad == 0 and nc <= target_k:
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_colors = cur.copy()
                break
    
    if best_colors is None:
        # Fallback to seed
        best_colors = np.array([(i+j) % 17 for i,j in edges], dtype=np.int32)
    
    return best_colors.tolist()
