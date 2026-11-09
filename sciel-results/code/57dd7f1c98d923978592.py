import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_np = np.array(primes)
    
    L_max = 10000
    
    # State
    coverage = np.zeros(L_max + 1, dtype=np.int16)
    residues = np.array([np.random.randint(0, p) for p in primes])
    
    def apply_change(p_idx, new_r):
        old_r = residues[p_idx]
        if old_r == new_r:
            return
        p = primes[p_idx]
        coverage[old_r::p] -= 1
        residues[p_idx] = new_r
        coverage[new_r::p] += 1
        
    def get_score():
        # Returns largest L such that 1..L are covered >= 2 times
        # If coverage[i] < 2 for some i <= L_max, L is the previous index.
        # If all 1..L_max are covered, return L_max.
        # We only care about indices 1..L_max.
        # Find first index in 1..L_max where coverage < 2.
        # If none, return L_max.
        # Using argmax on boolean mask
        # Note: coverage[0] is not used for score (score starts at 1).
        # But coverage[0] might be < 2. We ignore it.
        
        # Check if any failure in 1..L_max
        # If L_max is 0, return 0
        if L_max == 0:
            return 0
            
        # Create mask for 1..L_max
        # To avoid copying, we can just check the slice
        # np.argmax returns the first True. If no True, returns 0.
        # But we need to distinguish "no failure" from "failure at 0".
        # Since we look at 1..L_max, index 0 of the slice is index 1 of coverage.
        
        # Optimization: 
        # If we know the current best score, we only need to check from best_score+1?
        # No, changes can break things before best_score.
        # So we must check from 1.
        
        # However, we can early exit if coverage[1] < 2.
        # 
        # Let's just use np.where.
        fails = np.where(coverage[1:L_max+1] < 2)[0]
        if len(fails) == 0:
            return L_max
        return fails[0] # This is the count of covered integers before the first hole.
        # Example: hole at index 5 (in slice), means coverage[6] < 2.
        # Integers 1..5 are covered. Score = 5.
        # fails[0] is 4 (0-indexed in slice).
        # Wait.
        # Slice is coverage[1], coverage[2], ...
        # Index 0 in slice is coverage[1].
        # If coverage[1] < 2, fails[0] = 0. Score = 0. Correct.
        # If coverage[1]>=2, coverage[2]<2, fails[0]=1. Score = 1. Correct.
        
    # Initialize coverage
    for i, p in enumerate(primes):
        r = residues[i]
        coverage[r::p] += 1
        
    best_score = get_score()
    best_residues = residues.copy()
    
    # Greedy Pre-processing
    for _ in range(2000):
        if time.time() > deadline:
            break
        score = get_score()
        if score >= L_max:
            break
            
        fails = np.where(coverage[1:L_max+1] < 2)[0]
        if len(fails) == 0:
            break
            
        hole_idx = fails[0] + 1 # Actual integer value
        
        # Find primes that do NOT cover hole_idx
        # r_p != hole_idx % p
        target_res = hole_idx % primes_np
        mask = (residues != target_res)
        candidates = np.where(mask)[0]
        
        if len(candidates) == 0:
            # Should not happen if score < L_max
            break
            
        # Pick a candidate
        # Prefer primes with smaller p? Or random?
        # Random is fine.
        p_idx = candidates[np.random.randint(len(candidates))]
        new_r = hole_idx % primes[p_idx]
        
        apply_change(p_idx, new_r)
        
        new_score = get_score()
        if new_score > best_score:
            best_score = new_score
            best_residues = residues.copy()
            
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    no_improve_counter = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if T < T_min or no_improve_counter > max_no_improve:
            break
            
        move_type = np.random.rand()
        
        if move_type < 0.6:
            # Greedy Repair
            fails = np.where(coverage[1:L_max+1] < 2)[0]
            if len(fails) == 0:
                # No holes, try random moves to explore
                pass
            else:
                hole_idx = fails[0] + 1
                target_res = hole_idx % primes_np
                mask = (residues != target_res)
                candidates = np.where(mask)[0]
                if len(candidates) > 0:
                    p_idx = candidates[np.random.randint(len(candidates))]
                    new_r = hole_idx % primes[p_idx]
                    
                    old_r = residues[p_idx]
                    if old_r != new_r:
                        apply_change(p_idx, new_r)
                        new_score = get_score()
                        delta = new_score - best_score
                        
                        if delta > 0 or np.random.rand() < np.exp(delta / T):
                            if new_score > best_score:
                                best_score = new_score
                                best_residues = residues.copy()
                                no_improve_counter = 0
                            else:
                                no_improve_counter += 1
                        else:
                            # Revert
                            apply_change(p_idx, old_r)
                            no_improve_counter += 1
                        T *= cooling
                        continue
                        
        elif move_type < 0.9:
            # Random Flip
            p_idx = np.random.randint(n_primes)
            old_r = residues[p_idx]
            new_r = np.random.randint(0, primes[p_idx])
            
            if old_r != new_r:
                apply_change(p_idx, new_r)
                new_score = get_score()
                delta = new_score - best_score
                
                if delta > 0 or np.random.rand() < np.exp(delta / T):
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                else:
                    # Revert
                    apply_change(p_idx, old_r)
                    no_improve_counter += 1
                T *= cooling
                continue
        else:
            # Swap
            p_idx1 = np.random.randint(n_primes)
            p_idx2 = np.random.randint(n_primes)
            if p_idx1 == p_idx2:
                continue
                
            r1 = residues[p_idx1]
            r2 = residues[p_idx2]
            
            if r1 != r2:
                # Swap
                apply_change(p_idx1, r2)
                apply_change(p_idx2, r1)
                
                new_score = get_score()
                delta = new_score - best_score
                
                if delta > 0 or np.random.rand() < np.exp(delta / T):
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                else:
                    # Revert
                    apply_change(p_idx1, r1)
                    apply_change(p_idx2, r2)
                    no_improve_counter += 1
                T *= cooling
                continue
                
        # If we reached here, it means a move wasn't applied or was skipped
        # In the greedy branch, if no candidates or no holes, we fall through?
        # No, the greedy branch has `continue` inside.
        # If no holes, we fall through to end of loop?
        # We should handle the "no holes" case in greedy by doing a random move or just cooling.
        # Let's just cool and continue if no move was made.
        T *= cooling
        no_improve_counter += 1

    return list(best_residues)
