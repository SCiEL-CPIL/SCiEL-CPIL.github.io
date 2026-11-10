"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Target n=34 as it is the known maximum and yields highest score
    target_n = 34
    
    # Fallback if 34 fails, but we prioritize 34.
    # We will just run the search for 34. If it fails, we return the best found 
    # (which might be < 34 if we implement a fallback loop, but let's focus on 34 first).
    
    # To be safe and potentially score higher if 34 is hard, we could try 35? 
    # No, R(4,5;3) > 34 is the published bound. It's likely 34 is the max or close.
    # Actually, if R(4,5;3) is the threshold, finding a coloring for n=34 proves R(4,5;3) > 34.
    # The task says "Maximize n". If 34 is the max, score is 34/9.
    
    def solve_for_n(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        # Map K4/K5 to triple indices
        k4_triples = []
        for k4 in k4s:
            ts = []
            for i in range(4):
                t = tuple(sorted(k4[:i] + k4[i+1:]))
                ts.append(triple_to_idx[t])
            k4_triples.append(ts)
            
        k5_triples = []
        for k5 in k5s:
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[i] for i in combo]))
                ts.append(triple_to_idx[t])
            k5_triples.append(ts)
            
        # Reverse maps: triple index -> list of k4 indices, triple index -> list of k5 indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        for k_idx, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4s[t_idx].append(k_idx)
                
        triple_to_k5s = [[] for _ in range(num_triples)]
        for k_idx, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5s[t_idx].append(k_idx)

        # State variables
        # colors: 0 for Red, 1 for Blue
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        # Counts
        red_count_k4 = np.zeros(num_k4, dtype=np.int8)
        blue_count_k5 = np.zeros(num_k5, dtype=np.int8)
        
        # Initialize counts
        for k_idx, ts in enumerate(k4_triples):
            for t_idx in ts:
                if colors[t_idx] == 0: # Red
                    red_count_k4[k_idx] += 1
                    
        for k_idx, ts in enumerate(k5_triples):
            for t_idx in ts:
                if colors[t_idx] == 1: # Blue
                    blue_count_k5[k_idx] += 1
                    
        # Initial violations
        cur_violations = np.sum(red_count_k4 == 4) + np.sum(blue_count_k5 == 10)
        
        # Local Search / Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        # Precompute list of triples for random selection
        all_triples = np.arange(num_triples)
        
        steps = 0
        max_steps = 500000 # Cap steps to avoid infinite loop if time is not checked
        
        while time.time() < deadline - 5.0 and steps < max_steps:
            # Pick random triple
            t_idx = np.random.choice(all_triples)
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_v = 0
            
            # Update Red K4 counts
            for k_idx in triple_to_k4s[t_idx]:
                if old_color == 0: # Red -> Blue
                    # red_count decreases
                    if red_count_k4[k_idx] == 4:
                        # Was violation, now not
                        delta_v -= 1
                    # Update count
                    red_count_k4[k_idx] -= 1
                else: # Blue -> Red
                    # red_count increases
                    if red_count_k4[k_idx] == 3:
                        # Was ok, now violation
                        delta_v += 1
                    # Update count
                    red_count_k4[k_idx] += 1

            # Update Blue K5 counts
            for k_idx in triple_to_k5s[t_idx]:
                if old_color == 0: # Red -> Blue
                    # blue_count increases
                    if blue_count_k5[k_idx] == 9:
                        # Was ok, now violation
                        delta_v += 1
                    # Update count
                    blue_count_k5[k_idx] += 1
                else: # Blue -> Red
                    # blue_count decreases
                    if blue_count_k5[k_idx] == 10:
                        # Was violation, now not
                        delta_v -= 1
                    # Update count
                    blue_count_k5[k_idx] -= 1
            
            # Accept move
            if delta_v < 0:
                # Always accept improving moves
                colors[t_idx] = new_color
                cur_violations += delta_v
                if cur_violations == 0:
                    return n, colors
            else:
                # Accept with probability exp(-delta_v / T)
                if T > min_T:
                    prob = np.exp(-delta_v / T)
                    if np.random.random() < prob:
                        colors[t_idx] = new_color
                        cur_violations += delta_v
                        if cur_violations == 0:
                            return n, colors
                # If rejected, we already updated the counts in the logic above?
                # NO! The logic above updated the counts unconditionally.
                # We must revert if rejected.
                # This is inefficient. Better to calculate delta, then update.
                # But calculating delta requires checking the state.
                # Let's re-structure: Calculate delta first, then update.
                
                # REVERT COUNTS
                for k_idx in triple_to_k4s[t_idx]:
                    if old_color == 0: # We did Red->Blue (decremented)
                        red_count_k4[k_idx] += 1 # Revert
                    else: # We did Blue->Red (incremented)
                        red_count_k4[k_idx] -= 1 # Revert
                        
                for k_idx in triple_to_k5s[t_idx]:
                    if old_color == 0: # We did Red->Blue (incremented)
                        blue_count_k5[k_idx] -= 1 # Revert
                    else: # We did Blue->Red (decremented)
                        blue_count_k5[k_idx] += 1 # Revert

            T *= cooling
            steps += 1
            
            # Restart logic: if stuck for a long time, randomize?
            # Simple restart: if T is low and violations > 0, restart with new random init?
            # Or just keep going.
            
        return n, colors # Return best effort (likely not 0 violations)

    # Try to find a valid coloring for n=34
    # We might need multiple restarts if the first one fails
    best_n = 0
    best_colors = None
    
    # Since 34 is the target, we spend most time there.
    # If we find a valid one, we return immediately.
    
    # Let's do a few restarts within the time limit
    restarts = 0
    max_restarts = 10
    
    while time.time() < deadline - 10.0 and restarts < max_restarts:
        n, colors = solve_for_n(target_n)
        # Check if valid
        # We need a fast validity check on the final colors
        # Re-count violations for the final colors to be sure
        triples = list(combinations(range(target_n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Rebuild k4/k5 structures? No, we can just check the specific counts if we stored them?
        # The solve_for_n returns colors. We need to verify.
        # It's faster to just re-run the count logic on the returned colors.
        
        # Actually, solve_for_n returns n, colors.
        # We can just check if it's valid by re-computing violations.
        
        # To save time, let's assume if solve_for_n returned early with 0 violations, it's valid.
        # But if it returned due to timeout, it might not be.
        # Let's add a flag or check.
        
        # Let's modify solve_for_n to return (n, colors, is_valid)
        # For now, let's just re-verify.
        
        # Re-verification
        k4s = list(combinations(range(target_n), 4))
        k5s = list(combinations(range(target_n), 5))
        
        # This is expensive. 
        # Alternative: Trust the internal counter if it hit 0.
        # If it didn't hit 0, it's invalid.
        
        # Let's refine solve_for_n to return validity flag.
        
        break # The current implementation doesn't return validity flag cleanly.
              # Let's rewrite solve_for_n to be cleaner.

    # Rewrite solve_for_n for clarity and correctness
    def solve_for_n_v2(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        k4_triples = []
        for k4 in k4s:
            ts = []
            for i in range(4):
                t = tuple(sorted(k4[:i] + k4[i+1:]))
                ts.append(triple_to_idx[t])
            k4_triples.append(ts)
            
        k5_triples = []
        for k5 in k5s:
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[i] for i in combo]))
                ts.append(triple_to_idx[t])
            k5_triples.append(ts)
            
        triple_to_k4s = [[] for _ in range(num_triples)]
        for k_idx, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4s[t_idx].append(k_idx)
                
        triple_to_k5s = [[] for _ in range(num_triples)]
        for k_idx, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5s[t_idx].append(k_idx)

        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        red_count_k4 = np.zeros(num_k4, dtype=np.int8)
        blue_count_k5 = np.zeros(num_k5, dtype=np.int8)
        
        for k_idx, ts in enumerate(k4_triples):
            for t_idx in ts:
                if colors[t_idx] == 0:
                    red_count_k4[k_idx] += 1
                    
        for k_idx, ts in enumerate(k5_triples):
            for t_idx in ts:
                if colors[t_idx] == 1:
                    blue_count_k5[k_idx] += 1
                    
        cur_violations = np.sum(red_count_k4 == 4) + np.sum(blue_count_k5 == 10)
        
        if cur_violations == 0:
            return n, colors, True
            
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        all_triples = np.arange(num_triples)
        
        steps = 0
        max_steps = 1000000
        
        while time.time() < deadline - 5.0 and steps < max_steps:
            t_idx = np.random.choice(all_triples)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_v = 0
            
            # Calculate delta and update counts
            # We update counts, and if rejected, we revert.
            
            # Red K4
            for k_idx in triple_to_k4s[t_idx]:
                if old_color == 0: # Red -> Blue
                    if red_count_k4[k_idx] == 4:
                        delta_v -= 1
                    red_count_k4[k_idx] -= 1
                else: # Blue -> Red
                    if red_count_k4[k_idx] == 3:
                        delta_v += 1
                    red_count_k4[k_idx] += 1

            # Blue K5
            for k_idx in triple_to_k5s[t_idx]:
                if old_color == 0: # Red -> Blue
                    if blue_count_k5[k_idx] == 9:
                        delta_v += 1
                    blue_count_k5[k_idx] += 1
                else: # Blue -> Red
                    if blue_count_k5[k_idx] == 10:
                        delta_v -= 1
                    blue_count_k5[k_idx] -= 1
            
            accepted = False
            if delta_v < 0:
                accepted = True
            else:
                if T > min_T:
                    prob = np.exp(-delta_v / T)
                    if np.random.random() < prob:
                        accepted = True
            
            if accepted:
                colors[t_idx] = new_color
                cur_violations += delta_v
                if cur_violations == 0:
                    return n, colors, True
            else:
                # Revert counts
                for k_idx in triple_to_k4s[t_idx]:
                    if old_color == 0:
                        red_count_k4[k_idx] += 1
                    else:
                        red_count_k4[k_idx] -= 1
                for k_idx in triple_to_k5s[t_idx]:
                    if old_color == 0:
                        blue_count_k5[k_idx] -= 1
                    else:
                        blue_count_k5[k_idx] += 1

            T *= cooling
            steps += 1
            
        return n, colors, False

    # Run search
    # Try n=34
    for _ in range(5): # Max 5 restarts
        if time.time() > deadline - 10.0:
            break
        n, colors, is_valid = solve_for_n_v2(34)
        if is_valid:
            return {'n': n, 'colors': colors.tolist()}
            
    # If 34 fails, try 33, 32...
    for n in range(33, 9, -1):
        if time.time() > deadline - 5.0:
            break
        n_res, colors, is_valid = solve_for_n_v2(n)
        if is_valid:
            return {'n': n_res, 'colors': colors.tolist()}
            
    # Fallback: return the trivial n=9 solution
    from itertools import combinations
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

# EVOLVE-BLOCK-END
