import numpy as np
import time
import random

def construct():
    start_time = time.time()
    best_c = None
    best_score = 100  # Max possible is 23, but start high
    
    def get_score(c):
        # c is a numpy array of 12 integers
        # Compute square
        conv = np.convolve(c, c)
        return np.count_nonzero(conv)

    def local_search(c_init, max_time=55):
        c = c_init.copy()
        current_score = get_score(c)
        
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            for idx in range(12):
                old_val = c[idx]
                # Try a range of values around old_val and some randoms
                # We want to try values that might cause cancellation.
                # Small integers are most likely to create exact cancellations with other small integers.
                # Range -10 to 10 is a good heuristic.
                candidates = list(range(-10, 11))
                if old_val not in candidates:
                    candidates.append(old_val)
                
                for new_val in candidates:
                    if new_val == old_val:
                        continue
                    if new_val == 0:
                        continue
                    if abs(new_val) > 64:
                        continue
                    
                    c[idx] = new_val
                    new_score = get_score(c)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        c[idx] = old_val
                        # If we found a better score, we restart the loop from idx 0? 
                        # Or just continue? Continuing is faster but might miss global improvements.
                        # Restarting is safer but slower. Given 12 dims, restarting is cheap.
                        # Let's just break and restart the outer loop if we improved.
                        # Actually, just setting improved=True and breaking inner loop is efficient.
                        # But we need to reset idx to 0? 
                        # Standard Hill Climbing: scan all, if any improvement, repeat scan.
                        # This is "first improvement" or "best improvement". 
                        # Let's use "best improvement" within the scan for robustness.
                        pass
            
            # Optimization: If we improved, we should restart the scan from 0 
            # to ensure we don't miss better moves earlier in the array.
            # The loop above continues to the next idx. 
            # Let's change structure: 
            # while improved:
            #   improved = False
            #   best_delta = 0
            #   best_idx = -1
            #   best_val = None
            #   for idx in range(12):
            #      ... find best move for this idx ...
            #   if best_idx != -1:
            #      apply best move
            #      improved = True
            
        return c, current_score

    # Refined Local Search
    def hill_climb(c_init, max_time=55):
        c = c_init.copy()
        current_score = get_score(c)
        
        while time.time() - start_time < max_time:
            best_delta = 0
            best_idx = -1
            best_val = None
            
            for idx in range(12):
                old_val = c[idx]
                # Check a set of candidate values
                # Focus on small integers as they are more likely to cancel
                candidates = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]
                if old_val not in candidates:
                    candidates.append(old_val)
                
                for new_val in candidates:
                    if new_val == old_val or new_val == 0:
                        continue
                    if abs(new_val) > 64:
                        continue
                    
                    c[idx] = new_val
                    new_score = get_score(c)
                    delta = current_score - new_score
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_idx = idx
                        best_val = new_val
                    
                    c[idx] = old_val
            
            if best_idx != -1:
                c[best_idx] = best_val
                current_score -= best_delta
            else:
                break
                
        return c, current_score

    # Generate diverse starting points
    starting_points = []
    
    # 1. Alternating signs
    for sign in [1, -1]:
        starting_points.append(np.array([sign if j % 2 == 0 else -sign for j in range(12)], dtype=int))
        
    # 2. Binomial coefficients (smaller ones)
    # 1, 11, 55... too big. Use smaller patterns.
    # 1, -1, 1, -1... is already covered.
    
    # 3. Random small integers
    for _ in range(20):
        if time.time() - start_time > 50:
            break
        c_init = np.array([random.choice([-2, -1, 1, 2]) for _ in range(12)], dtype=int)
        # Ensure no zeros (already handled by choice)
        starting_points.append(c_init)
        
    # 4. Random from wider range
    for _ in range(20):
        if time.time() - start_time > 50:
            break
        c_init = np.array([random.randint(-5, 5) for _ in range(12)], dtype=int)
        # Replace zeros with 1
        c_init[c_init == 0] = 1
        starting_points.append(c_init)
        
    # 5. Structured: Powers of 2? No, too big. 
    # 6. Structured: 1, 0, 0... not allowed (must be 12 non-zero).
    
    # Run hill climbing on each starting point
    for c_init in starting_points:
        if time.time() - start_time > 55:
            break
            
        c_res, score_res = hill_climb(c_init, max_time=55)
        
        if score_res < best_score:
            best_score = score_res
            best_c = c_res.copy()
            
        # If we found a very low score, stop early?
        # Theoretical min for n=12 is likely around 10-15?
        # 23 is the upper bound for AP exponents.
        if best_score <= 12:
            break

    # Fallback
    if best_c is None:
        best_c = np.array([1]*12, dtype=int)
        
    return [int(x) for x in best_c]
