"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 30
    # LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    L = reduce(lcm, range(1, n + 1))
    
    # Weights w[i] such that sum c[i] * w[i] is the numerator
    # c[i] / (i+1) = c[i] * (L/(i+1)) / L
    w = [L // (i + 1) for i in range(n)]
    
    def get_value(coeffs):
        """Calculate the signed sum numerator."""
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * w[i]
        return s

    def is_valid(coeffs):
        return all(c in (-1, 0, 1) for c in coeffs)

    # Strategy 1: Greedy construction
    # We want to minimize |sum c[i] w[i]| != 0.
    # Start with all 0. 
    # A good heuristic for small subset sum is to pick large weights to "balance" or create small residuals.
    # However, we are free to choose -1, 0, 1.
    
    # Let's try a simple greedy:
    # Sort indices by weight descending.
    # At each step, decide c[i] to minimize the potential for future correction?
    # Actually, a standard greedy for "make sum close to 0" is:
    # 1. Start with empty set.
    # 2. Add largest weight.
    # 3. If sum > 0, try to subtract next largest?
    
    # Better: Randomized Greedy with Local Search.
    
    best_coeffs = [0] * n
    best_val = float('inf')
    
    # Fallback valid solution (e.g., 1/29 - 1/30? No, 1/1 - 1/2 - 1/3 ... )
    # Simple valid: [1, -1, 0, ..., 0] -> 1 - 1/2 = 1/2. Num = L/2.
    # Or [0, ..., 0, 1, -1] -> 1/29 - 1/30 = 1/(29*30). Num = L/870.
    fallback = [0] * n
    fallback[28] = 1
    fallback[29] = -1
    best_coeffs = fallback[:]
    best_val = abs(get_value(fallback))
    
    # Local Search / Hill Climbing
    # Neighbor: flip one coefficient, or swap two?
    # Flipping c[i] from 0 to 1 changes sum by w[i].
    # Flipping c[i] from 1 to 0 changes sum by -w[i].
    # Flipping c[i] from 1 to -1 changes sum by -2*w[i].
    
    def local_search(start_coeffs, time_limit=10.0):
        coeffs = start_coeffs[:]
        current_val = get_value(coeffs)
        current_abs = abs(current_val)
        
        if current_abs == 0:
            # Need nonzero. If we hit 0, we are in a "hole". 
            # We should avoid 0 or jump out of it.
            # For now, if we hit 0, we consider it bad for the "minimize nonzero" goal 
            # unless we can't find anything else. But the task says "must be nonzero".
            # So 0 is invalid.
            pass

        improved = True
        while improved and (time.time() < start_time + time_limit):
            improved = False
            # Try flipping each coefficient
            best_neighbor = None
            best_neighbor_val = current_abs
            best_neighbor_coeff = None
            
            # We want to minimize abs(sum) but sum != 0.
            # If current is 0, we want to make it 1 (or -1).
            # If current is non-zero, we want to reduce its magnitude.
            
            # Heuristic: 
            # If current_abs == 0:
            #   Look for a move that makes it small non-zero.
            # Else:
            #   Look for a move that makes it smaller.
            
            for i in range(n):
                c = coeffs[i]
                for nc in (-1, 0, 1):
                    if nc == c:
                        continue
                    # Apply change
                    coeffs[i] = nc
                    new_val = get_value(coeffs)
                    new_abs = abs(new_val)
                    
                    # Validity: new_val != 0
                    if new_val == 0:
                        # Invalid for final answer, but might be a good intermediate?
                        # Actually, if we are at 0, we MUST move away.
                        # If we are non-zero, moving to 0 is bad because we can't stop there.
                        # But maybe moving to 0 allows a subsequent move to 1?
                        # Let's treat 0 as "very large" for the purpose of local search 
                        # unless we are stuck at 0.
                        
                        # If current is 0, new_val 0 is bad (no progress).
                        # If current is non-zero, new_val 0 is a "trap".
                        # Let's just ignore moves to 0 for now to keep it simple, 
                        # unless current is 0.
                        
                        if current_val != 0:
                            continue # Don't go to 0 if we are non-zero
                        else:
                            # If we are at 0, we must move. 
                            # But this loop is for finding neighbors.
                            # If we are at 0, any non-zero move is an improvement over "invalid".
                            # But we start non-zero.
                            pass

                    # Compare
                    # We want to minimize abs(new_val) subject to new_val != 0.
                    # If current_val == 0 (shouldn't happen in valid search), we want smallest non-zero.
                    # If current_val != 0, we want new_abs < current_abs.
                    
                    if current_val == 0:
                        if new_val != 0 and new_abs < best_neighbor_val:
                            best_neighbor_val = new_abs
                            best_neighbor = i
                            best_neighbor_coeff = nc
                    else:
                        if new_val != 0 and new_abs < current_abs:
                            best_neighbor_val = new_abs
                            best_neighbor = i
                            best_neighbor_coeff = nc
                            
            if best_neighbor is not None:
                coeffs[best_neighbor] = best_neighbor_coeff
                current_val = get_value(coeffs)
                current_abs = abs(current_val)
                improved = True
            else:
                # Stuck. Try a random kick?
                # Pick a random index and flip it.
                if time.time() < start_time + time_limit - 1:
                    i = random.randint(0, n - 1)
                    c = coeffs[i]
                    # Pick a different value
                    options = [x for x in (-1, 0, 1) if x != c]
                    nc = random.choice(options)
                    coeffs[i] = nc
                    current_val = get_value(coeffs)
                    current_abs = abs(current_val)
                    # Continue loop to see if we can climb out
                    improved = True # Force one more iteration
                    # To avoid infinite loops in kick, we can limit kicks, 
                    # but the time limit handles it.
                    
        return coeffs, abs(get_value(coeffs))
    
    # Run multiple random starts
    num_starts = 200
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Generate random start
        # Bias towards having some non-zeros
        start_coeffs = [random.choice((-1, 0, 1)) for _ in range(n)]
        
        # Ensure it's not zero sum initially? 
        # Local search handles it, but if it starts at 0, it will kick.
        
        coeffs, val = local_search(start_coeffs, time_limit=0.2) # Short time per start
        
        if val < best_val and val > 0:
            best_val = val
            best_coeffs = coeffs[:]
            
        # Also try greedy start:
        # Sort by weight, pick signs to minimize partial sum?
        # Simple greedy: 
        # g = [0]*n
        # for i in sorted(range(n), key=lambda x: -w[x]):
        #    # try to add -1, 0, 1 to keep sum small
        #    # This is basically what local search does, but from a specific start
        
    # Final check
    final_val = get_value(best_coeffs)
    if final_val == 0:
        # Fallback
        best_coeffs = fallback[:]
        
    return best_coeffs

# EVOLVE-BLOCK-END
