import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    def build_structure(n):
        triples = list(combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        
        sets_4 = list(combinations(range(n), 4))
        set4_triples = []
        for s in sets_4:
            ts = [triple_index[(s[0], s[1], s[2])], triple_index[(s[0], s[1], s[3])], 
                  triple_index[(s[0], s[2], s[3])], triple_index[(s[1], s[2], s[3])]]
            set4_triples.append(ts)
            
        sets_5 = list(combinations(range(n), 5))
        set5_triples = []
        for s in sets_5:
            ts = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        ts.append(triple_index[(s[i], s[j], s[k])])
            set5_triples.append(ts)
            
        num_triples = len(triples)
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
            'num_set4': len(sets_4),
            'num_set5': len(sets_5)
        }

    def init_counters(colors, struct):
        cnt0_s4 = np.zeros(struct['num_set4'], dtype=int)
        for ts in struct['set4_triples']:
            for t_idx in ts:
                if colors[t_idx] == 0:
                    cnt0_s4[struct['set4_triples'].index(ts)] += 1 # This is slow, let's do it properly in build or init loop
            
        # Better init:
        cnt0_s4 = np.zeros(struct['num_set4'], dtype=int)
        for i, ts in enumerate(struct['set4_triples']):
            c = 0
            for t_idx in ts:
                if colors[t_idx] == 0:
                    c += 1
            cnt0_s4[i] = c
            
        cnt1_s5 = np.zeros(struct['num_set5'], dtype=int)
        for i, ts in enumerate(struct['set5_triples']):
            c = 0
            for t_idx in ts:
                if colors[t_idx] == 1:
                    c += 1
            cnt1_s5[i] = c
            
        v4 = np.sum(cnt0_s4 == 4)
        v5 = np.sum(cnt1_s5 == 10)
        return cnt0_s4, cnt1_s5, v4 + v5

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # We will try to find valid colorings for increasing N
    # Start from a reasonable N to save time on trivial small N
    # But we need a valid base to extend? No, we can just search for N directly.
    # However, searching for N=30 from scratch is hard.
    # Strategy: Search for N=10, if found, extend to 11, etc.
    # Or: Search for N=20, if found, extend to 21...
    
    # Let's try to jump to N=20 first?
    # Actually, the current code starts at 10. Let's keep that but make the search much faster.
    
    current_n = 10
    max_n_to_try = 35
    
    struct = build_structure(current_n)
    colors = np.random.randint(0, 2, size=struct['num_triples']).astype(np.int8)
    
    valid_colors_prev = None
    
    while time.time() < deadline - 5.0:
        if valid_colors_prev is not None:
            current_n += 1
            if current_n > max_n_to_try:
                break
            
            # Extend
            new_struct = build_structure(current_n)
            new_colors = np.zeros(new_struct['num_triples'], dtype=np.int8)
            
            # Map old triples to new
            # Old triples are those with max vertex < current_n - 1
            old_n = current_n - 1
            old_struct = build_structure(old_n) # Rebuild to get indices? 
            # This is inefficient. Let's just do it manually.
            
            # Actually, let's just restart search for new N with random init if extension is complex.
            # But extension is better.
            # Let's use a dict for old triples.
            old_triple_map = {t: i for i, t in enumerate(old_struct['triples'])}
            
            for i, t in enumerate(new_struct['triples']):
                if t[2] < old_n:
                    new_colors[i] = valid_colors_prev[old_triple_map[t]]
                else:
                    new_colors[i] = random.randint(0, 1)
                    
            struct = new_struct
            colors = new_colors
            valid_colors_prev = None # Reset flag
            
        # Run SA for current_n
        cnt0_s4, cnt1_s5, current_viol = init_counters(colors, struct)
        
        if current_viol == 0:
            best_n = current_n
            best_colors = colors.tolist()
            valid_colors_prev = colors.copy()
            continue # Try next N
            
        T = 0.1
        no_improve = 0
        max_steps = 50000 # Increased steps
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
                
            t_idx = random.randint(0, struct['num_triples'] - 1)
            new_color = 1 - colors[t_idx]
            
            # Calculate Delta
            delta = 0
            
            # 4-sets
            for s4_idx in struct['triple_to_set4'][t_idx]:
                old_cnt = cnt0_s4[s4_idx]
                if new_color == 0:
                    # Becomes 0, so cnt0 increases
                    # Wait, if new_color is 0, it was 1. So it leaves the '1' group, enters '0'.
                    # cnt0 increases by 1.
                    new_cnt = old_cnt + 1
                else:
                    # Becomes 1, so it was 0. cnt0 decreases by 1.
                    new_cnt = old_cnt - 1
                    
                viol_before = 1 if old_cnt == 4 else 0
                viol_after = 1 if new_cnt == 4 else 0
                delta += (viol_after - viol_before)
                
                # We don't update the counter yet until we accept
                # But we need to store the change? 
                # No, we just calculate delta. If accepted, we update.
                
            # 5-sets
            for s5_idx in struct['triple_to_set5'][t_idx]:
                old_cnt = cnt1_s5[s5_idx]
                if new_color == 1:
                    # Becomes 1, so cnt1 increases by 1
                    new_cnt = old_cnt + 1
                else:
                    # Becomes 0, so cnt1 decreases by 1
                    new_cnt = old_cnt - 1
                    
                viol_before = 1 if old_cnt == 10 else 0
                viol_after = 1 if new_cnt == 10 else 0
                delta += (viol_after - viol_before)
                
            # Acceptance
            if delta < 0:
                # Accept
                colors[t_idx] = new_color
                current_viol += delta
                no_improve = 0
                
                # Update counters
                for s4_idx in struct['triple_to_set4'][t_idx]:
                    if new_color == 0:
                        cnt0_s4[s4_idx] += 1
                    else:
                        cnt0_s4[s4_idx] -= 1
                        
                for s5_idx in struct['triple_to_set5'][t_idx]:
                    if new_color == 1:
                        cnt1_s5[s5_idx] += 1
                    else:
                        cnt1_s5[s5_idx] -= 1
                        
            elif random.random() < np.exp(-delta / T):
                # Accept worse
                colors[t_idx] = new_color
                current_viol += delta
                no_improve = 0
                
                for s4_idx in struct['triple_to_set4'][t_idx]:
                    if new_color == 0:
                        cnt0_s4[s4_idx] += 1
                    else:
                        cnt0_s4[s4_idx] -= 1
                        
                for s5_idx in struct['triple_to_set5'][t_idx]:
                    if new_color == 1:
                        cnt1_s5[s5_idx] += 1
                    else:
                        cnt1_s5[s5_idx] -= 1
            else:
                no_improve += 1
                
            if current_viol == 0:
                best_n = current_n
                best_colors = colors.tolist()
                valid_colors_prev = colors.copy()
                break
                
            if no_improve > 1000:
                # Perturb
                for _ in range(10):
                    t_idx = random.randint(0, struct['num_triples'] - 1)
                    new_color = 1 - colors[t_idx]
                    colors[t_idx] = new_color
                    # Update counters for perturbation
                    for s4_idx in struct['triple_to_set4'][t_idx]:
                        if new_color == 0:
                            cnt0_s4[s4_idx] += 1
                        else:
                            cnt0_s4[s4_idx] -= 1
                    for s5_idx in struct['triple_to_set5'][t_idx]:
                        if new_color == 1:
                            cnt1_s5[s5_idx] += 1
                        else:
                            cnt1_s5[s5_idx] -= 1
                            
                # Recalculate violations
                current_viol = np.sum(cnt0_s4 == 4) + np.sum(cnt1_s5 == 10)
                no_improve = 0
                T *= 0.5 # Cool down? Or reheat?
                if T < 0.01:
                    T = 0.1

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
