import time

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 85.0
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        x_min = n // 4 + 1
        found = None
        n_start = time.time()
        remaining = deadline - n_start
        n_budget = min(0.25, remaining / max(1, 256 - idx))
        
        for x in range(x_min, x_min + 5000):
            if time.time() > deadline or (time.time() - n_start) > n_budget:
                break
                
            A = 4 * x - n
            B = n * x
            
            k_min = B // A + 1
            k_max = (2 * B - 1) // A
            
            if k_min > k_max:
                continue
                
            B_sq = B * B
            
            for k in range(k_min, k_max + 1):
                U = k * A - B
                
                if B_sq % U != 0:
                    continue
                    
                V = B_sq // U
                
                if (V + B) % A != 0:
                    continue
                    
                z = (V + B) // A
                y = (U + B) // A
                
                if y <= x:
                    continue
                if z <= y:
                    continue
                if z > DENOM_BOUND:
                    continue
                    
                found = [x, y, z]
                break
                
            if found:
                break
                
        results[idx] = found
        
    return results
