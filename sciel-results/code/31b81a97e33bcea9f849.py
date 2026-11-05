import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 30
    
    # Weights w_i = 1/i as Fractions
    weights = [Fraction(1, i) for i in range(1, n + 1)]
    
    def calculate_sum(coeffs):
        total = Fraction(0, 1)
        for i in range(n):
            if coeffs[i] != 0:
                total += coeffs[i] * weights[i]
        return total
        
    def abs_frac(f):
        if f < 0:
            return -f
        return f

    best_score = 0
    best_coeffs = None
    best_sum = None
    
    # Generate initial candidates
    # 1. Random starts
    # 2. Structured starts (alternating, blocks, etc.)
    
    starts = []
    
    # Add some structured starts
    # All zeros (will be forced to move)
    starts.append([0] * n)
    
    # Alternating 1, -1
    alt = [0] * n
    for i in range(n):
        alt[i] = 1 if i % 2 == 0 else -1
    starts.append(alt)
    
    # Alternating 1, 0
    alt0 = [0] * n
    for i in range(n):
        alt0[i] = 1 if i % 2 == 0 else 0
    starts.append(alt0)
    
    # Block patterns
    for block_size in [2, 3, 5, 10]:
        pat = [0] * n
        for i in range(n):
            pat[i] = 1 if (i // block_size) % 2 == 0 else -1
        starts.append(pat)
        
    # Add random starts
    # We can add more random starts if we have time
    num_random_starts = 50
    for _ in range(num_random_starts):
        rand_start = [random.choice([-1, 0, 1]) for _ in range(n)]
        starts.append(rand_start)

    # Local search
    for init_coeffs in starts:
        if time.time() - start_time > deadline:
            break
            
        current_coeffs = init_coeffs[:]
        current_sum = calculate_sum(current_coeffs)
        
        # If sum is zero, we need to make it non-zero.
        # Find the flip that results in the smallest non-zero absolute sum.
        if current_sum == 0:
            best_flip_val = None
            best_flip_idx = None
            min_abs_val = None
            
            for i in range(n):
                for new_val in [-1, 0, 1]:
                    if new_val == current_coeffs[i]:
                        continue
                    # Calculate new sum
                    delta = (new_val - current_coeffs[i]) * weights[i]
                    new_sum = current_sum + delta
                    if new_sum == 0:
                        continue
                    abs_val = abs_frac(new_sum)
                    if min_abs_val is None or abs_val < min_abs_val:
                        min_abs_val = abs_val
                        best_flip_val = new_val
                        best_flip_idx = i
            
            if best_flip_idx is not None:
                current_coeffs[best_flip_idx] = best_flip_val
                current_sum = calculate_sum(current_coeffs)
            else:
                # This shouldn't happen unless n=0
                continue

        # Now current_sum is non-zero. Perform hill climbing.
        improved = True
        while improved:
            if time.time() - start_time > deadline:
                break
                
            improved = False
            best_next_sum = current_sum
            best_next_coeffs = current_coeffs
            best_abs = abs_frac(current_sum)
            
            for i in range(n):
                for new_val in [-1, 0, 1]:
                    if new_val == current_coeffs[i]:
                        continue
                    # Calculate new sum
                    delta = (new_val - current_coeffs[i]) * weights[i]
                    new_sum = current_sum + delta
                    
                    if new_sum == 0:
                        # We must avoid zero. 
                        # If current is non-zero, moving to zero is bad.
                        # But maybe we can move to zero then to something small? 
                        # The problem requires the FINAL sum to be non-zero.
                        # It's safer to avoid zero states entirely in the search path 
                        # unless we are in a "forced" state, but we started non-zero.
                        # Actually, passing through zero might be necessary?
                        # No, if we are at S != 0, and we flip to 0, we lose the "non-zero" constraint 
                        # for that intermediate state. But the constraint is on the RETURNED value.
                        # However, if we return a value, it must be non-zero.
                        # If we step on zero, we can't return it. We must step off.
                        # But hill climbing usually assumes monotonic improvement of the objective.
                        # Objective: minimize |S| subject to S != 0.
                        # If we go to 0, |S| becomes 0, which is "better" in magnitude, 
                        # but invalid. So we should treat 0 as a forbidden state.
                        continue
                    
                    abs_val = abs_frac(new_sum)
                    if abs_val < best_abs:
                        best_abs = abs_val
                        best_next_sum = new_sum
                        # We need to store the coeff change
                        # We can just apply it if it's the best so far in this pass?
                        # Or store it. Storing is safer to ensure we apply the BEST flip.
                        # But since we iterate all flips, we can just track the best one.
                        # However, applying one flip changes the state for the next flip.
                        # Standard hill climbing: evaluate all neighbors, pick the best one, apply it.
                        # So we need to store the best neighbor.
                        # To save memory/time, we can just store the index and value.
                        # Let's use a variable to track the best flip found in this iteration.
                        # We are already doing that implicitly by updating best_next_sum.
                        # But we need the coeffs.
                        # Let's just store the best (idx, val)
                        # Actually, the loop structure above doesn't store idx/val for the best.
                        # Let's refactor slightly.
                        pass # Placeholder for logic refinement

            # Refactored inner loop for clarity and correctness
            # Re-implementing the inner loop properly
            improved = False
            best_abs_current = abs_frac(current_sum)
            best_idx = -1
            best_val = -2 # Sentinel
            
            for i in range(n):
                for new_val in [-1, 0, 1]:
                    if new_val == current_coeffs[i]:
                        continue
                    delta = (new_val - current_coeffs[i]) * weights[i]
                    new_sum = current_sum + delta
                    
                    if new_sum == 0:
                        continue
                    
                    abs_val = abs_frac(new_sum)
                    if abs_val < best_abs_current:
                        best_abs_current = abs_val
                        best_idx = i
                        best_val = new_val
                        improved = True
            
            if improved:
                current_coeffs[best_idx] = best_val
                current_sum = calculate_sum(current_coeffs)
            else:
                break # No improvement found, local minimum reached

        # Check if this local minimum is the global best so far
        if current_sum != 0:
            curr_abs = abs_frac(current_sum)
            # Score is 1/curr_abs. We want to maximize score, so minimize curr_abs.
            if best_score == 0 or curr_abs < abs_frac(best_sum):
                best_score = 1 / curr_abs # Not strictly needed, just for comparison
                best_sum = current_sum
                best_coeffs = current_coeffs[:]
                
    # If no valid non-zero sum was found (highly unlikely), return a fallback
    if best_coeffs is None:
        # Fallback: c_1 = 1, others 0 -> Sum = 1
        fallback = [0] * n
        fallback[0] = 1
        return fallback
        
    return best_coeffs
