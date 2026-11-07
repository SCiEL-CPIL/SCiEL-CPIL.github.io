import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    all_masks = list(range(total_masks))
    
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    def count_comparable(fam):
        c = 0
        for i in range(len(fam)):
            for j in range(i + 1, len(fam)):
                if is_comparable(fam[i], fam[j]):
                    c += 1
        return c

    def run_local_search(fam):
        fam_set = set(fam)
        
        def get_gain(x, fam_list):
            c = 0
            for y in fam_list:
                if is_comparable(x, y):
                    c += 1
            return c

        improved = True
        # Limit iterations to avoid infinite loops if logic is flawed, though improved should become False
        max_iter = 1000
        iter_count = 0
        while improved and iter_count < max_iter:
            iter_count += 1
            improved = False
            for i in range(m):
                old_val = fam[i]
                fam_set.remove(old_val)
                F_minus_i = fam[:i] + fam[i+1:]
                old_gain = get_gain(old_val, F_minus_i)
                
                best_new_val = old_val
                best_gain = old_gain # If no better found, keep old
                
                for x in all_masks:
                    if x in fam_set:
                        continue
                    gain = get_gain(x, F_minus_i)
                    if gain > best_gain:
                        best_gain = gain
                        best_new_val = x
                
                if best_new_val != old_val:
                    fam[i] = best_new_val
                    fam_set.add(best_new_val)
                    improved = True
        
        return fam

    # Initialization
    # Use a seed based on time to vary, but keep it reproducible-ish? 
    # Actually, just use random.
    # If we want to be safe, we can try a few different random seeds quickly.
    
    best_fam = None
    best_score = -1
    
    # Try a few initializations
    # 1. Random
    # 2. Chain-based (optional, but random is usually fine)
    
    attempts = 5
    for k in range(attempts):
        if time.time() > deadline - 10.0:
            break
            
        random.seed(k * 12345 + 999)
        initial_fam = random.sample(all_masks, m)
        
        # Run local search
        # We might want to interrupt local search if time is running out
        # But local search is fast.
        current_fam = run_local_search(initial_fam)
        current_score = count_comparable(current_fam)
        
        if current_score > best_score:
            best_score = current_score
            best_fam = current_fam
            
        # If we have a very good score, we might stop early?
        # Max possible is 780.
        # If we get > 500, it's likely near optimal.
        if best_score > 600:
            break

    # If we still have time, do more random restarts
    while time.time() < deadline - 5.0:
        if best_fam is None:
             best_fam = random.sample(all_masks, m)
             best_score = count_comparable(best_fam)
             
        current_fam = random.sample(all_masks, m)
        current_fam = run_local_search(current_fam)
        current_score = count_comparable(current_fam)
        
        if current_score > best_score:
            best_score = current_score
            best_fam = current_fam

    # Convert to lists of ints
    return [[b for b in range(n) if (mask >> b) & 1] for mask in best_fam]
