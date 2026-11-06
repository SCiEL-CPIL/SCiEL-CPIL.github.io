import itertools
import random
import time
import math

def construct():
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    num_edges = len(edges)
    
    subsets = list(itertools.combinations(range(n), 4))
    num_subsets = len(subsets)
    
    edge_to_subsets = [[] for _ in range(num_edges)]
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(u, v)] = idx
        
    for s_idx, sub in enumerate(subsets):
        for u, v in itertools.combinations(sub, 2):
            e_idx = edge_index[(u, v)]
            edge_to_subsets[e_idx].append(s_idx)
            
    colors = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
        
    subset_counts = [{} for _ in range(num_subsets)]
    subset_distinct = [0] * num_subsets
    
    def init_state():
        for s_idx, sub in enumerate(subsets):
            cnt = {}
            for u, v in itertools.combinations(sub, 2):
                c = colors[edge_index[(u, v)]]
                cnt[c] = cnt.get(c, 0) + 1
            subset_counts[s_idx] = cnt
            subset_distinct[s_idx] = len(cnt)
            
    init_state()
    
    freq = {}
    for c in colors:
        freq[c] = freq.get(c, 0) + 1
        
    current_count = len(freq)
    
    def get_active_colors():
        return [c for c in freq if freq[c] > 0]
        
    def apply_move(e_idx, new_color):
        nonlocal current_count
        nonlocal freq
        
        old_color = colors[e_idx]
        if old_color == new_color:
            return True
            
        affected_subsets = edge_to_subsets[e_idx]
        
        # Check validity
        for s_idx in affected_subsets:
            cnt = subset_counts[s_idx]
            old_count = cnt.get(old_color, 0)
            new_count = cnt.get(new_color, 0)
            
            delta_distinct = 0
            if old_count == 1:
                delta_distinct -= 1
            if new_count == 0:
                delta_distinct += 1
                
            if subset_distinct[s_idx] + delta_distinct < 5:
                return False
                
        # Apply
        for s_idx in affected_subsets:
            cnt = subset_counts[s_idx]
            old_count = cnt.get(old_color, 0)
            new_count = cnt.get(new_color, 0)
            
            cnt[old_color] = old_count - 1
            if cnt[old_color] == 0:
                del cnt[old_color]
                
            cnt[new_color] = new_count + 1
            
            if old_count == 1:
                subset_distinct[s_idx] -= 1
            if new_count == 0:
                subset_distinct[s_idx] += 1
                
        # Update global
        freq[old_color] -= 1
        if freq[old_color] == 0:
            del freq[old_color]
            
        if new_color not in freq:
            freq[new_color] = 0
        freq[new_color] += 1
        
        # Update count
        if old_color not in freq: # Was removed
             current_count -= 1
        if new_color not in freq: # Was added (check before increment? No, we incremented above)
             # Actually, we check if new_color was 0 before increment.
             pass
             
        # Better way to update current_count:
        # If freq[old_color] becomes 0, count decreases.
        # If freq[new_color] was 0, count increases.
        # We already updated freq.
        # So:
        # if old_color not in freq: count -= 1
        # if new_color in freq and freq[new_color] == 1: count += 1 (Wait, if it was 0, it's now 1. So it was added)
        
        # Let's recalculate delta properly in the caller or here.
        # I will return the delta.
        
        colors[e_idx] = new_color
        return True

    # I will rewrite the loop to handle revert manually to avoid deep copies.
    
    best_count = current_count
    best_colors = colors.copy()
    
    start_time = time.time()
    time_limit = 115.0
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        e_idx = random.randint(0, num_edges - 1)
        current_c = colors[e_idx]
        
        active = [c for c in freq if freq[c] > 0]
        if len(active) <= 1:
            break
            
        new_c = random.choice(active)
        if new_c == current_c:
            continue
            
        # Save old state for revert
        old_color = current_c
        old_subset_states = []
        for s_idx in edge_to_subsets[e_idx]:
            # Store (color, old_count, new_count_was_zero, distinct_change)
            # Or just store the dict and distinct? Dict copy is expensive.
            # Store: old_count_old, new_count_old, distinct_old
            cnt = subset_counts[s_idx]
            oc = cnt.get(old_color, 0)
            nc = cnt.get(new_c, 0)
            d = subset_distinct[s_idx]
            old_subset_states.append((s_idx, oc, nc, d))
            
        old_freq_old = freq.get(old_color, 0)
        old_freq_new = freq.get(new_c, 0)
        old_count = current_count
        
        # Try move
        if apply_move(e_idx, new_c):
            # Calculate delta
            # delta = current_count - old_count
            # But apply_move doesn't return delta.
            # We can compute it:
            # If old_color count became 0, delta -= 1
            # If new_c count was 0, delta += 1
            
            delta = 0
            if old_freq_old == 1:
                delta -= 1
            if old_freq_new == 0:
                delta += 1
                
            accept = False
            if delta <= 0:
                accept = True
            else:
                if random.random() < math.exp(-delta / T):
                    accept = True
                    
            if not accept:
                # Revert
                colors[e_idx] = old_color
                for s_idx, oc, nc, d in old_subset_states:
                    cnt = subset_counts[s_idx]
                    # Revert counts
                    # Current state: old_color decreased, new_c increased.
                    # Revert: old_color increase, new_c decrease.
                    cnt[old_color] = cnt.get(old_color, 0) + 1
                    cnt[new_c] = cnt.get(new_c, 0) - 1
                    if cnt[new_c] == 0:
                        del cnt[new_c]
                    
                    # Revert distinct
                    # If oc == 1, distinct was decreased by 1. Now increase.
                    # If nc == 0, distinct was increased by 1. Now decrease.
                    if oc == 1:
                        subset_distinct[s_idx] += 1
                    if nc == 0:
                        subset_distinct[s_idx] -= 1
                    # Ensure it matches d?
                    # subset_distinct[s_idx] should be d.
                    # Let's just set it to d to be safe.
                    subset_distinct[s_idx] = d
                    
                # Revert freq
                freq[old_color] = old_freq_old
                if old_freq_new == 0:
                    if new_c in freq:
                        del freq[new_c]
                else:
                    freq[new_c] = old_freq_new
                    
                current_count = old_count
            else:
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
                    
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_colors
