"""Strategy: Greedy Descent with Exact Rational Arithmetic."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n = 30
    # We want to minimize |sum c_i / i| != 0
    # c_i in {-1, 0, 1}
    
    best_val = None
    best_coeffs = None
    best_num = None
    best_den = None
    
    # Initialize with a random sparse vector to ensure non-zero start
    # But we will search aggressively.
    # A good heuristic: Try to cancel the largest terms with combinations of smaller ones.
    # The sum of reciprocals 1/1 + ... + 1/30 is about 4.
    # We can form many small sums using the tail.
    
    # We will use a multi-start local search / greedy approach.
    # State: list of 30 ints in {-1, 0, 1}
    # Objective: minimize abs(sum c_i/i) subject to sum != 0
    
    def evaluate(coeffs):
        s = Fraction(0, 1)
        for i in range(n):
            if coeffs[i] != 0:
                s += Fraction(coeffs[i], i + 1)
        if s == 0:
            return None, None, None
        return abs(s), coeffs, s.numerator, s.denominator
    
    # Helper: Calculate sum quickly using precomputed denominators?
    # Fraction is robust but slow. For 30 terms it's fine.
    # LCM of 1..30 is 2329089562800. We can use integers for speed.
    # Sum = (sum c_i * (L/i)) / L
    # L = lcm(1, ..., 30)
    
    import math
    L = 1
    for i in range(1, n+1):
        L = L * i // math.gcd(L, i)
        
    # Precompute weights
    weights = [L // (i + 1) for i in range(n)]
    
    def evaluate_int(coeffs):
        s = 0
        for i in range(n):
            if coeffs[i] != 0:
                s += coeffs[i] * weights[i]
        if s == 0:
            return None
        return abs(s)
    
    def get_frac_from_int(s):
        # s is numerator over L
        g = math.gcd(abs(s), L)
        num = s // g
        den = L // g
        return num, den

    # Local Search / Greedy Step
    def local_search(coeffs, time_limit):
        # Greedy descent: try flipping each coefficient to -1, 0, 1
        # Pick the move that minimizes abs sum, tie-break by randomness or keep
        # If stuck, do a random perturbation
        
        current_coeffs = coeffs[:]
        current_val = evaluate_int(current_coeffs)
        if current_val is None:
            # If zero, we must move. Pick random non-zero move
            for _ in range(100):
                idx = random.randrange(n)
                old = current_coeffs[idx]
                new = random.choice([-1, 0, 1])
                if new != old:
                    current_coeffs[idx] = new
                    current_val = evaluate_int(current_coeffs)
                    if current_val is not None:
                        break
            if current_val is None:
                return current_coeffs, current_val
        
        best_val = current_val
        best_coeffs = current_coeffs[:]
        
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            # Try all single flips
            best_move_val = best_val
            best_move_idx = -1
            best_move_new_val = None
            
            # To speed up, we can sample if n is large, but n=30 is small.
            # 30 * 3 = 90 checks per iteration.
            for i in range(n):
                old_val = current_coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    # Calculate new sum efficiently?
                    # s_new = s_old - old_val * weights[i] + new_val * weights[i]
                    # We need the signed sum to check if it becomes 0 or smaller.
                    # Let's maintain signed sum for efficiency.
                    pass
            
            # Re-implement with signed sum for speed
            signed_sum = 0
            for i in range(n):
                if current_coeffs[i] != 0:
                    signed_sum += current_coeffs[i] * weights[i]
                    
            if signed_sum == 0:
                # Must escape zero
                # Random move
                idx = random.randrange(n)
                current_coeffs[idx] = random.choice([-1, 0, 1])
                signed_sum += (current_coeffs[idx] - (current_coeffs[idx] if False else 0)) # Wait, logic error in escape
                # Let's just re-evaluate after random move
                for _ in range(10):
                    idx = random.randrange(n)
                    old = current_coeffs[idx]
                    new = random.choice([-1, 0, 1])
                    if new != old:
                        current_coeffs[idx] = new
                        signed_sum += (new - old) * weights[idx]
                        if signed_sum != 0:
                            break
                if signed_sum == 0:
                     # Still zero, force a specific change
                     current_coeffs[0] = 1
                     signed_sum = evaluate_int(current_coeffs)
                     if signed_sum == 0: return current_coeffs, None
                     signed_sum = sum(c*w for c,w in zip(current_coeffs, weights))
                     
            best_abs = abs(signed_sum)
            best_coeffs_local = current_coeffs[:]
            
            # Greedy step
            for i in range(n):
                old_val = current_coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    new_signed = signed_sum - old_val * weights[i] + new_val * weights[i]
                    if new_signed == 0:
                        continue # We want non-zero
                    new_abs = abs(new_signed)
                    if new_abs < best_abs:
                        best_abs = new_abs
                        best_move_idx = i
                        best_move_new_val = new_val
                        best_coeffs_local = current_coeffs[:]
                        best_coeffs_local[i] = new_val
                        signed_sum = new_signed
                        improved = True
                        # Restart scan? Or continue? 
                        # Restarting is safer for greedy descent to ensure we don't miss a better move after a change
                        # But for n=30, full scan is cheap.
                        # Let's just break and restart the outer loop if we found improvement
                        break
                if improved:
                    break
            
            if improved:
                current_coeffs = best_coeffs_local
                # signed_sum is already updated to best_signed
            else:
                # Local minimum reached
                # Perturbation: flip 2-3 random coefficients
                k = random.randint(1, 3)
                for _ in range(k):
                    idx = random.randrange(n)
                    current_coeffs[idx] = random.choice([-1, 0, 1])
                # Recalculate signed_sum
                signed_sum = 0
                for i in range(n):
                    if current_coeffs[i] != 0:
                        signed_sum += current_coeffs[i] * weights[i]
                if signed_sum == 0:
                    # Escape zero again
                    idx = random.randrange(n)
                    current_coeffs[idx] = random.choice([-1, 0, 1])
                    signed_sum = sum(c*w for c,w in zip(current_coeffs, weights))
                    if signed_sum == 0:
                         current_coeffs[0] = 1
                         signed_sum = sum(c*w for c,w in zip(current_coeffs, weights))
                # Continue loop to try to descend from new point
                improved = True # Force at least one more attempt? Or just break?
                # Actually, if we perturbed, we should try to descend again.
                # But to avoid infinite loops, we can just break and rely on multi-start.
                break
        
        return current_coeffs, abs(signed_sum)

    # Multi-Start Search
    num_starts = 5000
    for start in range(num_starts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Bias towards sparse? Or random?
        # Random with p=0.3 for -1, 0.3 for 1, 0.4 for 0
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Ensure non-zero
        s_val = evaluate_int(coeffs)
        if s_val is None:
            coeffs[0] = 1
            s_val = evaluate_int(coeffs)
            if s_val is None:
                continue
                
        # Run local search with a small time budget per start
        # Since local search is fast, we can run it fully or with a step limit
        # Let's run a simplified local search: just a few greedy steps
        # The `local_search` above is a bit complex. Let's do a simpler hill climbing.
        
        current_coeffs = coeffs[:]
        current_signed = sum(c*w for c,w in zip(current_coeffs, weights))
        if current_signed == 0:
             # fix
             current_coeffs[0] = 1
             current_signed = sum(c*w for c,w in zip(current_coeffs, weights))
             if current_signed == 0:
                 continue
                 
        best_local_abs = abs(current_signed)
        
        # Greedy descent
        for step in range(100): # Max 100 steps per start
            if time.time() > deadline:
                break
            found_improvement = False
            # Try all moves
            for i in range(n):
                old_val = current_coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    new_signed = current_signed - old_val * weights[i] + new_val * weights[i]
                    if new_signed == 0:
                        continue
                    new_abs = abs(new_signed)
                    if new_abs < best_local_abs:
                        best_local_abs = new_abs
                        current_coeffs[i] = new_val
                        current_signed = new_signed
                        found_improvement = True
                        break
                if found_improvement:
                    break
            
            if not found_improvement:
                # Perturb
                # Flip 2 random elements
                for _ in range(2):
                    idx = random.randrange(n)
                    current_coeffs[idx] = random.choice([-1, 0, 1])
                current_signed = sum(c*w for c,w in zip(current_coeffs, weights))
                if current_signed == 0:
                    # Escape
                    idx = random.randrange(n)
                    current_coeffs[idx] = random.choice([-1, 0, 1])
                    current_signed = sum(c*w for c,w in zip(current_coeffs, weights))
                    if current_signed == 0:
                        # Give up on this start
                        break
                best_local_abs = abs(current_signed)
                # Continue trying to improve from perturbed state
                # But limit total steps to avoid getting stuck in one start too long
                # The outer loop handles diversity.
                break # Break inner loop, go to next start? 
                # Actually, if we perturb, we should try to descend again.
                # Let's allow a few more steps after perturbation.
                # But for simplicity and speed, let's just break and rely on next start.
                # Or, we can continue the `for step` loop.
                # Let's continue the step loop.
                pass 

        # End of local search for this start
        if best_local_abs is not None:
            if best_val is None or best_local_abs < best_val:
                best_val = best_local_abs
                best_coeffs = current_coeffs[:]
                # Calculate exact fraction
                # best_val is numerator over L
                num, den = get_frac_from_int(current_signed)
                # Note: current_signed might be negative, but we stored abs
                # We need the sign for the final report? 
                # "exact numerator and denominator are reported"
                # Usually numerator can be negative.
                # The task says "Minimize its absolute value".
                # So we report the fraction corresponding to the sum.
                # Let's store the signed sum.
                best_num = num
                best_den = den
                best_signed_sum = current_signed

    # If no valid result found (should not happen), return seed
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs
# EVOLVE-BLOCK-END
