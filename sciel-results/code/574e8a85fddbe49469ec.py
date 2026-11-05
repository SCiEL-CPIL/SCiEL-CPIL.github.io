"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    # Best known: [1]*12 has square with 13 nonzero terms
    best_square_count = 13
    best_poly = [1]*12
    
    # Verify baseline
    sq = np.convolve(np.array(best_poly, dtype=int), np.array(best_poly, dtype=int))
    best_square_count = int(np.count_nonzero(sq))
    
    # Strategy: Systematic search over structured coefficient patterns
    # Key idea: Use coefficients that create cancellation in the convolution
    # 
    # Approach 1: Search over exponent sets with arithmetic progression + coefficient patterns
    # Approach 2: Local search starting from good candidates
    # Approach 3: Try specific algebraic forms
    
    # Helper function to count nonzero terms in square
    def count_nonzero_square(poly):
        p = np.array(poly, dtype=int)
        sq = np.convolve(p, p)
        return int(np.count_nonzero(sq))
    
    # Try various structured patterns
    
    # Pattern 1: Alternating signs with various step sizes
    for step in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30]:
        if time.time() > deadline:
            break
        
        max_exp = 40 // step
        if max_exp < 11:
            continue
        
        # Try different sign patterns
        for sign_pattern in ['alternating', 'blocks', 'random']:
            if time.time() > deadline:
                break
            
            for trial in range(500):
                if time.time() > deadline:
                    break
                
                # Generate 12 distinct exponents that are multiples of step
                if max_exp + 1 >= 12:
                    exponents = sorted(random.sample(range(max_exp + 1), 12))
                    exponents = [e * step for e in exponents]
                else:
                    continue
                
                # Assign coefficients based on pattern
                coeffs = []
                for i, e in enumerate(exponents):
                    if sign_pattern == 'alternating':
                        coeffs.append(1 if i % 2 == 0 else -1)
                    elif sign_pattern == 'blocks':
                        # Blocks of same sign
                        block = (i // 3) % 2
                        coeffs.append(1 if block == 0 else -1)
                    else:
                        coeffs.append(random.choice([-1, 1]))
                
                poly = [0] * 41
                for i, e in enumerate(exponents):
                    if e < 41:
                        poly[e] = coeffs[i]
                
                # Check exactly 12 nonzero terms
                if sum(1 for c in poly if c != 0) != 12:
                    continue
                
                nz = count_nonzero_square(poly)
                if nz < best_square_count:
                    best_square_count = nz
                    best_poly = poly[:]
    
    # Pattern 2: Local search / hill climbing
    # Start from a good candidate and try small perturbations
    if time.time() < deadline:
        # Try starting from various initial candidates
        initial_candidates = []
        
        # All ones
        cand1 = [0]*41
        for i in range(12):
            cand1[i] = 1
        initial_candidates.append(cand1)
        
        # Alternating
        cand2 = [0]*41
        for i in range(12):
            cand2[i] = 1 if i % 2 == 0 else -1
        initial_candidates.append(cand2)
        
        # Step 2
        cand3 = [0]*41
        for i in range(12):
            cand3[2*i] = 1
        initial_candidates.append(cand3)
        
        for init_poly in initial_candidates:
            if time.time() > deadline:
                break
            
            current = init_poly[:]
            current_nz = count_nonzero_square(current)
            
            # Hill climbing: try flipping signs, moving terms, changing coefficients
            for iteration in range(2000):
                if time.time() > deadline:
                    break
                
                # Pick a random operation
                op = random.choice(['flip_sign', 'move_term', 'change_coeff'])
                
                new_poly = current[:]
                changed = False
                
                if op == 'flip_sign':
                    # Flip sign of a random nonzero term
                    nonzero_indices = [i for i, c in enumerate(current) if c != 0]
                    if nonzero_indices:
                        idx = random.choice(nonzero_indices)
                        new_poly[idx] = -current[idx]
                        changed = True
                elif op == 'move_term':
                    # Move a term to a new position
                    nonzero_indices = [i for i, c in enumerate(current) if c != 0]
                    if nonzero_indices:
                        src = random.choice(nonzero_indices)
                        # Find a new position
                        zero_indices = [i for i, c in enumerate(current) if c == 0]
                        if zero_indices:
                            dst = random.choice(zero_indices)
                            new_poly[dst] = current[src]
                            new_poly[src] = 0
                            changed = True
                elif op == 'change_coeff':
                    # Change coefficient of a term
                    nonzero_indices = [i for i, c in enumerate(current) if c != 0]
                    if nonzero_indices:
                        idx = random.choice(nonzero_indices)
                        # Try small integer coefficients
                        new_coeff = random.choice([-2, -1, 1, 2])
                        if new_coeff != current[idx]:
                            new_poly[idx] = new_coeff
                            changed = True
                
                if not changed:
                    continue
                
                # Validate: exactly 12 nonzero terms, coefficients in [-64, 64]
                nz_count = sum(1 for c in new_poly if c != 0)
                if nz_count != 12:
                    continue
                if any(c < -64 or c > 64 for c in new_poly):
                    continue
                if new_poly[-1] == 0:
                    # Trim trailing zeros
                    while len(new_poly) > 1 and new_poly[-1] == 0:
                        new_poly.pop()
                    if len(new_poly) > 41:
                        continue
                
                new_nz = count_nonzero_square(new_poly)
                if new_nz < current_nz:
                    current = new_poly
                    current_nz = new_nz
                    if current_nz < best_square_count:
                        best_square_count = current_nz
                        best_poly = current[:]
    
    # Pattern 3: Try specific algebraic identities
    if time.time() < deadline:
        # P(x) = (1 + x^k) * Q(x) where Q has 11 terms
        # This might create structure that helps cancellation
        
        for k in range(1, 41):
            if time.time() > deadline:
                break
            
            # Try P(x) = (1 + x^k) * (sum of 11 terms)
            # The square will have cross terms that might cancel
            
            # Generate random 11-term polynomial
            for trial in range(200):
                if time.time() > deadline:
                    break
                
                # 11 distinct exponents, none equal to 0 or k (to avoid conflicts)
                available = [i for i in range(41) if i != 0 and i != k and i + k <= 40]
                if len(available) < 11:
                    continue
                
                exponents = sorted(random.sample(available, 11))
                coeffs = [random.choice([-1, 1]) for _ in range(11)]
                
                # Build P(x) = (1 + x^k) * Q(x)
                poly = [0] * 41
                # Add Q(x) terms
                for i, e in enumerate(exponents):
                    poly[e] = coeffs[i]
                # Add x^k * Q(x) terms
                for i, e in enumerate(exponents):
                    if e + k < 41:
                        poly[e + k] += coeffs[i]
                
                # Check exactly 12 nonzero terms
                nz_count = sum(1 for c in poly if c != 0)
                if nz_count != 12:
                    continue
                
                if any(c < -64 or c > 64 for c in poly):
                    continue
                
                nz = count_nonzero_square(poly)
                if nz < best_square_count:
                    best_square_count = nz
                    best_poly = poly[:]
    
    # Final validation
    final_sq = np.convolve(np.array(best_poly, dtype=int), np.array(best_poly, dtype=int))
    final_nz = int(np.count_nonzero(final_sq))
    
    # Ensure valid output
    result = best_poly[:]
    # Trim trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    # Pad to at least length 1
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
