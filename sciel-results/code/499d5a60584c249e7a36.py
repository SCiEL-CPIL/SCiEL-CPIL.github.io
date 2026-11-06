"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def count_nonzero(arr):
    return sum(1 for x in arr if x != 0)

def compute_square_nonzero(poly):
    square = np.convolve(poly, poly)
    return count_nonzero(square)

def construct():
    start_time = time.time()
    best_poly = None
    best_nonzero = float('inf')
    
    # Strategy: Local Search on Coefficients for structured Exponents
    # This approach focuses on finding cancellation patterns by optimizing
    # coefficients for fixed exponent sets, which is more directed than pure random sampling.
    
    structures = []
    
    # 1. Arithmetic Progressions (High collision probability)
    for step in range(1, 12):
        for start in range(0, 5):
            exponents = [start + i * step for i in range(12)]
            if max(exponents) <= 40:
                structures.append(exponents)
                
    # 2. Random distinct exponents (Diversity)
    for _ in range(30):
        exponents = random.sample(range(0, 41), 12)
        structures.append(sorted(exponents))

    # Limit structures to ensure time efficiency
    if len(structures) > 40:
        structures = structures[:40]

    for exponents in structures:
        if time.time() - start_time > 55:
            break
        
        max_deg = max(exponents)
        
        # Try multiple random initializations for coefficients
        for init_trial in range(10):
            if time.time() - start_time > 55:
                break
            
            # Initialize coefficients: mostly +/-1
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            
            current_poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                current_poly[e] = c
            
            current_nz = compute_square_nonzero(current_poly)
            
            # Hill Climbing
            improved = True
            while improved and time.time() - start_time < 55:
                improved = False
                # Try changing each coefficient
                for i in range(12):
                    old_c = coeffs[i]
                    # Try neighbors in value space
                    # Focus on small integers to maintain sparsity
                    candidates = [old_c - 1, old_c + 1, -old_c]
                    # Add a random jump occasionally
                    if random.random() < 0.1:
                        candidates.append(random.randint(-3, 3))
                    
                    # Filter valid candidates (non-zero, within bounds, distinct)
                    valid_cands = [c for c in candidates if c != 0 and -64 <= c <= 64 and c != old_c]
                    
                    if not valid_cands:
                        continue
                        
                    random.shuffle(valid_cands)
                    
                    for new_c in valid_cands[:2]: # Limit tries per index
                        coeffs[i] = new_c
                        current_poly[exponents[i]] = new_c
                        
                        new_nz = compute_square_nonzero(current_poly)
                        
                        if new_nz < current_nz:
                            current_nz = new_nz
                            improved = True
                            if current_nz < best_nonzero:
                                best_nonzero = current_nz
                                best_poly = current_poly[:]
                            # Accept and break to re-evaluate from new state
                            break 
                        else:
                            # Revert
                            coeffs[i] = old_c
                            current_poly[exponents[i]] = old_c

            # Update best if this local minimum is better
            if current_nz < best_nonzero:
                best_nonzero = current_nz
                best_poly = current_poly[:]

    # Fallback if nothing found (should not happen)
    if best_poly is None:
        best_poly = [0]*41
        for i in range(12):
            best_poly[i] = 1
            
    # Trim trailing zeros
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    # Ensure length <= 41
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
    
    # Ensure nonzero final coefficient
    if best_poly[-1] == 0:
        best_poly[-1] = 1
        
    return best_poly
# EVOLVE-BLOCK-END
