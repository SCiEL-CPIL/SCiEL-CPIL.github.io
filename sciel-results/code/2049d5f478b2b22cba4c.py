def construct():
    deadline = time.time() + 55
    n = 256
    rng = random.Random(42)
    
    best_signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
    best_val = _max_ap_discrepancy_fast(best_signs)
    
    while time.time() < deadline - 5:
        current_signs = best_signs.copy()
        current_val = best_val
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            best_flip = -1
            best_flip_val = current_val
            
            for i in range(n):
                current_signs[i] = -current_signs[i]
                v = _max_ap_discrepancy_fast(current_signs, limit=best_flip_val)
                if v < best_flip_val:
                    best_flip_val = v
                    best_flip = i
                current_signs[i] = -current_signs[i]
            
            if best_flip != -1:
                current_signs[best_flip] = -current_signs[best_flip]
                current_val = best_flip_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_signs = current_signs.copy()
        
        if time.time() < deadline - 10:
             signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
             val = _max_ap_discrepancy_fast(signs)
             if val < best_val:
                 best_signs = signs
                 best_val = val
                 
    return [int(x) for x in best_signs]
