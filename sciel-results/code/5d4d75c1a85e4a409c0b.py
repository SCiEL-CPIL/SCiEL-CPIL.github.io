import random
from fractions import Fraction
import time

def construct():
    n = 30
    # Precompute unit fractions
    invs = [Fraction(1, i+1) for i in range(n)]
    
    best_val = None
    best_coeffs = None
    
    # Time limit: 50 seconds to be safe
    start_time = time.time()
    time_limit = 50.0
    
    # Strategy: Random restarts with aggressive local search
    # We will try to find a configuration with sum close to 0.
    
    # To speed up, we can use a beam of best candidates or just many restarts.
    # Given 30 variables, local search is effective if we have good restarts.
    
    # Heuristic: Start with coefficients that balance the large terms.
    # e.g., c_1 = 1, c_2 = -1, c_3 = 1, etc?
    # Or just random.
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        
        # Initialize with random coefficients
        # Bias towards 0? Or uniform?
        # Uniform is fine.
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        current_sum = sum(c * inv for c, inv in zip(coeffs, invs))
        
        if current_sum == 0:
            # If zero, we must make it nonzero.
            # Flip a small coefficient to make it nonzero but small.
            # Find an index where we can change c to make sum nonzero.
            # Change c_i to c_i + 1 or -1.
            # To keep it small, pick a large i (small weight).
            # Try changing the last non-zero or just last index.
            for i in range(n-1, -1, -1):
                old_c = coeffs[i]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c: continue
                    coeffs[i] = new_c
                    new_sum = current_sum + (new_c - old_c) * invs[i]
                    if new_sum != 0:
                        current_sum = new_sum
                        break
                if current_sum != 0:
                    break
            if current_sum == 0:
                # Fallback: set last coeff to 1
                coeffs[-1] = 1
                current_sum = sum(c * inv for c, inv in zip(coeffs, invs))
        
        # Local Search
        improved = True
        # Limit local search steps per restart to avoid getting stuck in one bad config
        max_steps = 1000
        steps = 0
        while improved and steps < max_steps and time.time() - start_time < time_limit:
            improved = False
            steps += 1
            
            # Try changing one coefficient
            # To speed up, we can shuffle the order of indices
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                old_c = coeffs[i]
                # Try other values
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    
                    delta = (new_c - old_c) * invs[i]
                    new_sum = current_sum + delta
                    
                    if new_sum == 0:
                        continue
                    
                    # Check if this is better than global best
                    if best_val is None or abs(new_sum) < abs(best_val):
                        # Accept this move as the new current state?
                        # Or just update global best?
                        # Hill climbing: accept if it improves current state.
                        # But we want global min.
                        # If new_sum is better than current_sum, accept it as new current.
                        if abs(new_sum) < abs(current_sum):
                            coeffs[i] = new_c
                            current_sum = new_sum
                            best_val = new_sum
                            best_coeffs = coeffs[:]
                            improved = True
                            break # Move to next i? Or restart loop?
                            # Restarting the loop allows us to try other moves from new state
                            # But breaking inner loop and continuing outer while is fine.
                        elif best_val is None or abs(new_sum) < abs(best_val):
                            # It doesn't improve current, but improves global best?
                            # This can happen if we are stuck in a local min for current,
                            # but a different branch leads to global min?
                            # No, if it doesn't improve current, we shouldn't move there
                            # in a simple hill climb.
                            # However, we can update best_coeffs if this specific
                            # single flip from current state is better than best found so far.
                            # This is a "best improvement" strategy but only looking 1-step ahead.
                            if best_val is None or abs(new_sum) < abs(best_val):
                                best_val = new_sum
                                best_coeffs = coeffs[:]
                                # Note: we do NOT change coeffs[i] here because it's not better than current.
                                # We just record it as a candidate.
                                # But wait, if we don't move there, we might miss a path.
                                # For now, let's stick to strict hill climbing for speed.
                                pass
            
            # If we broke out of i loop, we made a move.
            # Otherwise, no move improved current_sum.
        
        # After local search, we have a local minimum.
        # Check if it's the global best.
        if best_val is None or abs(current_sum) < abs(best_val):
            best_val = current_sum
            best_coeffs = coeffs[:]
            
        # Occasionally, try a "kick" or random perturbation if stuck?
        # The restarts handle that.

    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
