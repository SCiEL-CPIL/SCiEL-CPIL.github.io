"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit (or 90s if that's the hard kill, but 55s is safe for both if we assume 60s evaluation window usually, but prompt says 90s wall clock. Let's use 80s to be safe but fast.)
    # Actually, prompt says "evaluator KILLS the program after 90s wall-clock". 
    # "Each evaluation must finish within 60 s" is in the task description. 
    # Let's aim for 50s to be safe.
    deadline = start_time + 50.0

    n = 30
    # LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # We want to minimize |sum(c_i * L/i)| != 0
    # c_i in {-1, 0, 1}
    
    best_val = None
    best_coefs = None
    
    # Strategy: Randomized Greedy with Backtracking/Lookahead? 
    # Simple Greedy: At each step, choose c_i in {-1, 0, 1} that minimizes |current_sum + c_i * L/i|.
    # But we must ensure final sum is non-zero.
    # Greedy might land on 0. 
    # If it lands on 0, we can flip one sign or add a small perturbation.
    
    # Let's do multiple random restarts with different random initializations or random tie-breaking.
    
    def greedy_construct(seed_val):
        random.seed(seed_val)
        current_sum = 0
        coefs = [0] * n
        # Process terms in random order? Or fixed order?
        # Fixed order 1..n is standard. Random order might escape local minima of the greedy choice?
        # Let's stick to fixed order but random tie-breaking and random initial "bias".
        
        # Actually, a better greedy:
        # Sort indices by magnitude of L/i? No, L/i is just L/i.
        # L/1 is huge. L/30 is small.
        # Greedy usually works better if you handle large terms first to set the base, then fine-tune with small terms.
        # So order 1, 2, ..., 30 is good.
        
        for i in range(1, n + 1):
            val = L // i
            # Options: -val, 0, val
            # We want to minimize |current_sum + option|
            # But if we end up at 0, it's invalid.
            # We can track the best non-zero path? That's exponential.
            # Instead: just minimize |current_sum + option|.
            # If final is 0, we have a problem.
            
            # To avoid 0, we can add a small random noise or bias.
            # Or, we can simply check at the end. If 0, we modify the last few coefficients.
            
            options = [-val, 0, val]
            # Add slight randomization to break ties or explore
            # Sort options by resulting absolute sum
            scored_options = []
            for opt in options:
                new_sum = current_sum + opt
                # Tie-breaking: random
                scored_options.append((abs(new_sum), random.random(), opt))
            
            scored_options.sort()
            best_opt = scored_options[0][2]
            coefs[i-1] = best_opt
            current_sum += best_opt
            
        # Check if zero
        if current_sum == 0:
            # Try to fix by flipping a coefficient
            # Find an index where flipping changes sum to non-zero and minimizes new abs sum
            # Flipping c_i to -c_i changes sum by -2*c_i*L/i
            # If c_i is 0, flipping to 1 or -1 adds/subtracts L/i.
            
            min_new_val = None
            best_fix_idx = None
            best_fix_val = None
            
            for i in range(n):
                c = coefs[i]
                val = L // (i+1)
                
                # Try all possible changes for this coefficient
                for new_c in [-1, 0, 1]:
                    if new_c == c:
                        continue
                    # Change in sum = (new_c - c) * val
                    delta = (new_c - c) * val
                    new_sum = current_sum + delta
                    if new_sum == 0:
                        continue
                    if min_new_val is None or abs(new_sum) < min_new_val:
                        min_new_val = abs(new_sum)
                        best_fix_idx = i
                        best_fix_val = new_c
                        
            if best_fix_idx is not None:
                coefs[best_fix_idx] = best_fix_val
                current_sum += (best_fix_val - coefs[best_fix_idx]) * (L // (best_fix_idx + 1)) 
                # Wait, I already updated coefs in the loop above? No, I'm in a check phase.
                # Let's re-calculate sum properly if we fix it.
                # Actually, easier: just return the coefs and let the caller verify/fix?
                # No, let's just do a local search from the zero solution.
                
        return coefs, current_sum

    # Run multiple greedy constructions with different seeds
    for s in range(1000):
        if time.time() > deadline:
            break
        
        coefs, s_val = greedy_construct(s)
        
        # If sum is 0, perform a quick local search to make it non-zero
        if s_val == 0:
            # Local search: flip one coefficient
            found = False
            for i in range(n):
                if time.time() > deadline:
                    break
                old_c = coefs[i]
                val = L // (i+1)
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * val
                    new_s = s_val + delta
                    if new_s != 0:
                        coefs[i] = new_c
                        s_val = new_s
                        found = True
                        break
                if found:
                    break
            if not found:
                # Should not happen, but if it does, skip
                continue

        # Now we have a non-zero sum s_val
        # Improve it with local search (flipping single coefficients)
        # Goal: minimize |s_val|
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                old_c = coefs[i]
                val = L // (i+1)
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * val
                    new_s = s_val + delta
                    if new_s == 0:
                        continue
                    if abs(new_s) < abs(s_val):
                        coefs[i] = new_c
                        s_val = new_s
                        improved = True
                        # Restart loop from beginning? Or continue?
                        # Restarting is safer for hill climbing.
                        break
                if improved:
                    break
                    
        # Update best
        if best_val is None or abs(s_val) < best_val:
            best_val = abs(s_val)
            best_coefs = coefs[:]
            
            # If we found 1 (the theoretical minimum for integer sums), we can stop
            if best_val == 1:
                break

    if best_coefs is None:
        # Fallback: return seed solution
        return [0] * 28 + [1, -1]
        
    return best_coefs
# EVOLVE-BLOCK-END
