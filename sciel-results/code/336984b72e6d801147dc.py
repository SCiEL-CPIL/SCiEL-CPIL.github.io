import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    best_W = float('inf')
    best_sol = None
    
    # We know W >= 9.
    # We scan L. The optimal L doesn't need to be huge.
    # The pattern of multiples repeats with period LCM(divisors).
    # LCM is large, but the local structure repeats often enough.
    # Scanning 1..5000 should be sufficient.
    
    start_time = time.time()
    
    for L in range(1, 10000):
        if time.time() - start_time > 50:
            break
            
        x = [((L + d - 1) // d) * d for d in divisors]
        
        # Check distinctness
        if len(set(x)) != n:
            continue
            
        # Check max value constraint (should be fine for small L)
        if max(x) > 1000000:
            continue
            
        W = max(x) - min(x)
        
        if W < best_W:
            best_W = W
            best_sol = x
            
        if best_W == 9:
            break
            
    return best_sol
