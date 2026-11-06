import time

DENOM_BOUND = 10**12

def construct():
    # Set a safety margin to avoid timeout. 
    # The evaluator kills at 90s. We aim to finish in < 10s.
    deadline = time.time() + 85.0
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        # x must be > n/4
        x_min = n // 4 + 1
        found = None
        n_start = time.time()
        # Allocate a small budget per n, but allow more if we are fast
        n_budget = min(0.1, (deadline - n_start) / max(1, 256 - idx))
        
        # Try x values starting from x_min
        # Solutions are typically found for small x
        for x in range(x_min, x_min + 1000):
            if time.time() > deadline or (time.time() - n_start) > n_budget:
                break
                
            A = 4 * x - n
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # Let U = Ay - B, V = Az - B. UV = B^2.
            # y = (U+B)/A, z = (V+B)/A
            # Conditions:
            # 1. U divides B^2
            # 2. (U+B) % A == 0
            # 3. y > x  => (U+B)/A > x => U > Ax - B
            # 4. z > y  => V > U => B^2/U > U => U^2 < B^2 => U < B
            # 5. z <= DENOM_BOUND
            
            lower_bound_U = A * x - B
            upper_bound_U = B - 1
            
            if lower_bound_U < 0:
                lower_bound_U = 0
                
            # Iterate U from lower_bound_U to upper_bound_U
            # To speed up, we can iterate divisors of B^2 instead of all integers.
            # But for small B, linear scan is fine.
            # B is at most 256 * (256/4 + 1000) ~ 256 * 1064 ~ 272k.
            # B^2 is ~ 7 * 10^10.
            # Linear scan of 272k is fast.
            
            for U in range(lower_bound_U + 1, upper_bound_U + 1):
                if B * B % U != 0:
                    continue
                    
                V = (B * B) // U
                
                if (V + B) % A != 0:
                    continue
                    
                z = (V + B) // A
                y = (U + B) // A
                
                # Check distinctness and bounds
                if y <= x:
                    continue
                if z <= y:
                    continue
                if z > DENOM_BOUND:
                    continue
                if y > DENOM_BOUND:
                    continue
                    
                found = [x, y, z]
                break
                
            if found:
                break
                
        results[idx] = found
        
    return results
