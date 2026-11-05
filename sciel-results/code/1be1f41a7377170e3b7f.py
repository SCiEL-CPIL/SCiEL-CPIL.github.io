import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def get_triples(n):
        return list(combinations(range(n), 3))

    def build_structure(n):
        """
        Builds data structures for fast vectorized updates.
        """
        triples = list(combinations(range(n), 3))
        tri_index = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        # 4-sets: For each 4-set, store the indices of the 4 triples it contains
        sets_4 = list(combinations(range(n), 4))
        set4_triples = []
        for s in sets_4:
            a, b, c, d = s
            ts = [
                tri_index[(a, b, c)],
                tri_index[(a, b, d)],
                tri_index[(a, c, d)],
                tri_index[(b, c, d)]
            ]
            set4_triples.append(np.array(ts, dtype=np.int32))
        num_set4 = len(sets_4)
        
        # 5-sets: For each 5-set, store the indices of the 10 triples it contains
        sets_5 = list(combinations(range(n), 5))
        set5_triples = []
        for s in sets_5:
            ts = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        ts.append(tri_index[(s[i], s[j], s[k])])
            set5_triples.append(np.array(ts, dtype=np.int32))
        num_set5 = len(sets_5)
        
        # Reverse mapping: For each triple, which 4-sets and 5-sets contain it?
        triple_to_set4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(set4_triples):
            for t_idx in ts:
                triple_to_set4[t_idx].append(i)
                
        triple_to_set5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(set5_triples):
            for t_idx in ts:
                triple_to_set5[t_idx].append(i)
                
        return {
            'n': n,
            'triples': triples,
            'set4_triples': set4_triples,
            'set5_triples': set5_triples,
            'triple_to_set4': triple_to_set4,
            'triple_to_set5': triple_to_set5,
            'num_triples': num_triples,
            'num_set4': num_set4,
            'num_set5': num_set5
        }

    def init_violations(colors, struct):
        """Initialize violation counts for 4-sets and 5-sets."""
        num_set4 = struct['num_set4']
        num_set5 = struct['num_set5']
        
        viol4 = np.zeros(num_set4, dtype=np.int32)
        for i, ts in enumerate(struct['set4_triples']):
            viol4[i] = 4 - np.sum(colors[ts])
            
        viol5 = np.zeros(num_set5, dtype=np.int32)
        for i, ts in enumerate(struct['set5_triples']):
            viol5[i] = np.sum(colors[ts])
            
        total_v = int(np.sum(viol4 == 4) + np.sum(viol5 == 10))
        
        return viol4, viol5, total_v

    def simulate_annealing(struct, time_budget, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        num_triples = struct['num_triples']
        
        # Initialize colors randomly
        colors = np.random.randint(0, 2, size=num_triples)
        viol4, viol5, total_v = init_violations(colors, struct)
        
        best_colors = colors.copy()
        best_v = total_v
        
        # SA Parameters
        T = 2.0
        T_min = 0.05
        alpha = 0.999
        
        start_sa = time.time()
        
        while time.time() - start_sa < time_budget:
            if best_v == 0:
                break
                
            # Pick a random triple to flip
            t_idx = np.random.randint(0, num_triples)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate delta in violations
            d_v = 0
            
            # 4-sets update
            s4_indices = struct['triple_to_set4'][t_idx]
            if s4_indices:
                s4_arr = np.array(s4_indices, dtype=np.int32)
                old_counts = viol4[s4_arr].copy()
                
                if old_color == 0:
                    new_counts = old_counts - 1
                else:
                    new_counts = old_counts + 1
                    
                old_viols = (old_counts == 4).astype(np.int32)
                new_viols = (new_counts == 4).astype(np.int32)
                
                d_v += np.sum(new_viols - old_viols)
                
                viol4[s4_arr] = new_counts
                
            # 5-sets update
            s5_indices = struct['triple_to_set5'][t_idx]
            if s5_indices:
                s5_arr = np.array(s5_indices, dtype=np.int32)
                old_counts = viol5[s5_arr].copy()
                
                if old_color == 1:
                    new_counts = old_counts - 1
                else:
                    new_counts = old_counts + 1
                    
                old_viols = (old_counts == 10).astype(np.int32)
                new_viols = (new_counts == 10).astype(np.int32)
                
                d_v += np.sum(new_viols - old_viols)
                
                viol5[s5_arr] = new_counts
            
            # Acceptance criterion
            if d_v < 0:
                colors[t_idx] = new_color
                total_v += d_v
            else:
                if T > T_min:
                    if d_v > 100 * T:
                        prob = 0.0
                    else:
                        prob = np.exp(-d_v / T)
                    if np.random.random() < prob:
                        colors[t_idx] = new_color
                        total_v += d_v
            
            if total_v < best_v:
                best_v = total_v
                best_colors = colors.copy()
                
            T *= alpha
            if T < T_min:
                break
                
        return best_colors, best_v

    best_n = 0
    best_colors = None
    
    # Targets sorted descending
    targets = [
        (35, 30.0),
        (34, 60.0),
        (33, 10.0),
        (32, 5.0),
        (31, 5.0)
    ]
    
    for target_n, time_budget in targets:
        if time.time() > deadline - 2:
            break
            
        # Build structure once for this n
        if time.time() > deadline - 5:
            break
        struct = build_structure(target_n)
        
        found_valid = False
        start_attempt = time.time()
        
        while time.time() - start_attempt < time_budget:
            if time.time() > deadline - 2:
                break
                
            seed = random.randint(0, 2**31)
            run_time = min(5.0, time_budget - (time.time() - start_attempt))
            if run_time < 0.5: break
            
            colors, v = simulate_annealing(struct, run_time, seed)
            
            if v == 0:
                if target_n > best_n:
                    best_n = target_n
                    best_colors = colors.tolist()
                found_valid = True
                break
        
        if found_valid:
            # If we found a valid coloring for this n, we can stop trying lower n?
            # No, we want MAX n. 
            # If we found 34, we still want to try 35 (which is already done in this loop order? 
            # Wait, my loop is 35, 34, 33...
            # So if I find 35, I can break the whole loop? 
            # Yes, because 35 is the highest in my list.
            # If I find 34, I continue to 33? No, 33 is lower. 
            # I should break if I find the highest target in my list?
            # Actually, just let it run. 
            # But if I find 35, I can break immediately.
            if target_n == 35:
                break

    # Fallback if nothing found
    if best_colors is None:
        if time.time() < deadline - 10:
             struct20 = build_structure(20)
             colors, v = simulate_annealing(struct20, 10.0, 42)
             if v == 0:
                 best_n = 20
                 best_colors = colors.tolist()
        
        if best_colors is None:
             triples9 = get_triples(9)
             best_n = 9
             best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples9]

    # Verify and return
    triples_final = get_triples(best_n)
    if len(best_colors) != len(triples_final):
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_final]
        best_n = 9

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"Found n={res['n']}")
