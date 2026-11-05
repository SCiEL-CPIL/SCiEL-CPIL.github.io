"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from fractions import Fraction
import time

def construct():
    n = 30
    # We want to minimize |sum c_i / i| for c_i in {-1, 0, 1}, sum != 0.
    
    def calc_sum(coeff):
        s = Fraction(0, 1)
        for i, c in enumerate(coeff):
            if c != 0:
                s += Fraction(c, i + 1)
        return s

    best_val = None
    best_coeff = None
    
    # Use a few random restarts to find a good local minimum
    start_time = time.time()
    limit = 55.0 # Leave 5s buffer
    
    # Initial guesses:
    # 1. Random
    # 2. Alternating signs (Harmonic series partial sums are small)
    # 3. Zeros except a few terms
    
    candidates = []
    
    # Generate a few diverse starting points
    # Start 1: Random
    c1 = [random.choice([-1, 0, 1]) for _ in range(n)]
    candidates.append(c1)
    
    # Start 2: Alternating 1, -1, 1, -1...
    c2 = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(c2)
    
    # Start 3: Alternating -1, 1...
    c3 = [-1 if i % 2 == 0 else 1 for i in range(n)]
    candidates.append(c3)
    
    # Start 4: Only last few terms active (often small sums)
    c4 = [0] * n
    c4[-1] = 1
    c4[-2] = -1
    candidates.append(c4)
    
    # Start 5: Random with bias towards 0
    c5 = [random.choice([-1, 0, 0, 1]) for _ in range(n)]
    candidates.append(c5)

    for start_coeff in candidates:
        if time.time() - start_time > limit:
            break
            
        curr_coeff = start_coeff[:]
        curr_sum = calc_sum(curr_coeff)
        
        # If sum is 0, we must move. Pick a random non-zero move.
        if curr_sum == 0:
            idx = random.randrange(n)
            curr_coeff[idx] = 1 if curr_coeff[idx] != 1 else -1
            curr_sum = calc_sum(curr_coeff)
            
        curr_abs = abs(curr_sum)
        
        if best_val is None or curr_abs < best_val:
            best_val = curr_abs
            best_coeff = curr_coeff[:]
            
        # Local Search Phase
        improved = True
        while improved and (time.time() - start_time < limit):
            improved = False
            # Try flipping each coefficient
            # Shuffle indices to break symmetry/tie-breaking bias
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time > limit:
                    break
                old_val = curr_coeff[i]
                # Possible new values: -1, 0, 1 (excluding current)
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    
                    # Calculate new sum exactly
                    # new_sum = curr_sum + (new_val - old_val) / (i+1)
                    delta = Fraction(new_val - old_val, i + 1)
                    new_sum = curr_sum + delta
                    
                    if new_sum == 0:
                        continue # Must be nonzero
                    
                    new_abs = abs(new_sum)
                    
                    if new_abs < curr_abs:
                        # Accept improvement
                        curr_coeff[i] = new_val
                        curr_sum = new_sum
                        curr_abs = new_abs
                        improved = True
                        
                        if best_val is None or curr_abs < best_val:
                            best_val = curr_abs
                            best_coeff = curr_coeff[:]
                        
                        break # Restart loop or continue? 
                              # Continuing with the same i might allow further flips at same i? 
                              # No, we tried all 3 values. 
                              # But maybe another i is better now. 
                              # Standard hill climbing: accept and restart scan or continue scan?
                              # Continue scan is faster.
                    # If we want to escape local minima, we could add random moves, 
                    # but for "minimize absolute value", strict improvement is safest first.
                    # Let's add a small probability of accepting a worse move if stuck? 
                    # No, keep it simple first.

    # Final verification
    if best_coeff is None:
        # Fallback: simple non-zero
        best_coeff = [0]*n
        best_coeff[0] = 1
        return best_coeff
        
    final_sum = calc_sum(best_coeff)
    if final_sum == 0:
        # Should not happen if logic is correct, but safety check
        best_coeff[0] = 1
        if calc_sum(best_coeff) == 0:
             best_coeff[1] = -1
             
    return best_coeff
# EVOLVE-BLOCK-END
