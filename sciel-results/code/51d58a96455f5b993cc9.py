import numpy as np
import random
import time
import math

def construct():
    start_time = time.time()
    best_c = None
    best_score = 100
    
    def get_score(c):
        conv = np.convolve(c, c)
        return np.count_nonzero(conv)

    def local_search(c, max_time):
        c = c.copy()
        sq = np.convolve(c, c).astype(int)
        current_score = np.count_nonzero(sq)
        
        while time.time() - start_time < max_time:
            improved = False
            # Try to zero out non-zero terms in sq
            # We iterate through k
            for k in range(23):
                if sq[k] == 0:
                    continue
                
                # Try to adjust c[i] for i in 0..11 such that i+j=k for some valid j
                for i in range(12):
                    j = k - i
                    if j < 0 or j > 11:
                        continue
                    
                    old_val = c[i]
                    
                    if i == j:
                        # S_k includes c_i^2
                        # New S_k = S_k - c_i^2 + v^2
                        # We want New S_k = 0 => v^2 = c_i^2 - S_k
                        target_sq = c[i]**2 - sq[k]
                        if target_sq < 0:
                            continue
                        v = int(math.isqrt(target_sq))
                        if v*v != target_sq:
                            continue
                        if v == 0:
                            continue
                        if abs(v) > 64:
                            continue
                        
                        # Apply change
                        c[i] = v
                        # Update sq
                        # S_k changes by (v^2 - old_val^2)
                        sq[k] += v*v - old_val*old_val
                        
                        # Check if other sq terms changed?
                        # c[i] contributes to S_m for m in [i, i+11]
                        # We only checked S_k. Other S_m are now stale.
                        # We need to recalculate the whole sq or update all affected.
                        # For simplicity and correctness, let's just recalculate score if this looks promising
                        # But updating all affected is better.
                        
                        # Let's just recalculate the full square for simplicity in this thought process, 
                        # but in code we should be efficient.
                        # Given N=12, full recalc is cheap.
                        new_sq = np.convolve(c, c)
                        new_score = np.count_nonzero(new_sq)
                        
                        if new_score < current_score:
                            current_score = new_score
                            sq = new_sq
                            improved = True
                        else:
                            c[i] = old_val
                            # sq is not modified in the 'else' branch effectively, 
                            # but we modified sq[k] above. We need to restore it.
                            sq[k] -= v*v - old_val*old_val
                            # Actually, since we reverted c[i], sq should be back to original.
                            # But we modified sq[k] in place. 
                            # It's safer to just recompute sq if we revert? 
                            # Or just keep sq in sync.
                            # Since we only modified sq[k] and then reverted c[i], 
                            # the other sq entries were not touched in the update logic above 
                            # (I only updated sq[k]). 
                            # So sq is consistent with the reverted c[i] ONLY IF we didn't 
                            # rely on the intermediate sq state for other decisions.
                            # Since we break out of the inner loop on improvement, 
                            # and revert on failure, sq is consistent with c.
                            pass
                    else:
                        # S_k includes c_i * c_j
                        # New S_k = S_k - c_i*c_j + v*c_j
                        # We want New S_k = 0 => v*c_j = c_i*c_j - S_k
                        # v = c_i - S_k / c_j
                        if c[j] == 0:
                            continue
                        if sq[k] % c[j] != 0:
                            continue
                        v = c[i] - sq[k] // c[j]
                        if v == 0:
                            continue
                        if abs(v) > 64:
                            continue
                        
                        c[i] = v
                        # Update sq
                        delta = (v - old_val) * c[j]
                        sq[k] += delta
                        
                        new_sq = np.convolve(c, c)
                        new_score = np.count_nonzero(new_sq)
                        
                        if new_score < current_score:
                            current_score = new_score
                            sq = new_sq
                            improved = True
                        else:
                            c[i] = old_val
                            sq[k] -= delta
                            
            if not improved:
                # Try random perturbations to escape local minimum?
                # Or just break and let ILS handle it.
                break
                
        return c, current_score

    # ILS
    for restart in range(50):
        if time.time() - start_time > 55:
            break
            
        # Initialize
        # Use a mix of patterns
        if restart % 3 == 0:
            c_init = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        elif restart % 3 == 1:
            c_init = [1 if i % 2 == 0 else -1 for i in range(12)]
        else:
            c_init = [random.randint(-5, 5) for _ in range(12)]
            c_init = [1 if x == 0 else x for x in c_init]
            
        c_res, score_res = local_search(np.array(c_init, dtype=int), max_time=55)
        
        if best_c is None or score_res < best_score:
            best_score = score_res
            best_c = c_res.copy()
            
        # Perturb best solution for next iteration if we have time
        if best_c is not None and time.time() - start_time < 50:
            c_perturb = best_c.copy()
            # Change 2 random coefficients
            for _ in range(2):
                idx = random.randint(0, 11)
                c_perturb[idx] = random.randint(-10, 10)
                if c_perturb[idx] == 0:
                    c_perturb[idx] = 1
            # Run a short local search on perturbed
            c_res2, score_res2 = local_search(c_perturb, max_time=min(55, time.time() - start_time + 1))
            if score_res2 < best_score:
                best_score = score_res2
                best_c = c_res2.copy()

    if best_c is None:
        best_c = np.array([1]*12, dtype=int)
        
    return [int(x) for x in best_c]
