"""Rewritten program with a different search strategy: greedy construction + simulated annealing."""
import time
import numpy as np
from itertools import product, combinations

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Fast square computation using numpy convolution with exact integers
    def square_nonzero_count(poly):
        """Compute the number of nonzero terms in poly^2 using exact integer convolution."""
        # Trim to actual length
        n = len(poly)
        # Use numpy convolve for speed (exact for integers)
        sq = np.convolve(poly, poly)
        return int(np.count_nonzero(sq))
    
    def make_poly_from_exponents(exponents, coeffs):
        """Create a coefficient list from exponents and coefficients."""
        if not exponents:
            return [0]
        max_deg = max(exponents)
        poly = [0] * (max_deg + 1)
        for e, c in zip(exponents, coeffs):
            poly[e] = c
        return poly
    
    def trim(poly):
        """Remove trailing zeros."""
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        return poly
    
    def is_valid(poly):
        """Check all constraints."""
        if len(poly) > 41:
            return False
        if len(poly) < 1 or poly[-1] == 0:
            return False
        nonzero = sum(1 for c in poly if c != 0)
        if nonzero != 12:
            return False
        for c in poly:
            if c < -64 or c > 64:
                return False
        return True
    
    # Initial best: the trivial (1+x+...+x^11)^2
    best_poly = [1] * 12
    best_count = square_nonzero_count(best_poly)  # 23
    
    # Strategy 1: Greedy construction
    # Build the polynomial term by term, choosing each new term to minimize the square's nonzero count
    # while maintaining exactly 12 nonzero terms.
    
    # Start with a single term
    # We'll build up to 12 terms
    
    def greedy_build():
        """Build polynomial greedily, adding one term at a time."""
        # Start with [1] at position 0
        exponents = [0]
        coeffs = [1]
        
        for target_terms in range(2, 13):
            best_new_exp = None
            best_new_coeff = None
            best_score = float('inf')
            
            # Try all possible new positions and coefficients
            max_deg = max(exponents) + 20 if target_terms < 13 else max(exponents) + 10
            
            # For efficiency, limit the search space
            # Try positions near existing ones and some spread
            candidate_positions = set()
            for e in exponents:
                for delta in range(-10, 11):
                    pos = e + delta
                    if 0 <= pos <= 40:
                        candidate_positions.add(pos)
            
            # Also try some structured positions
            for d in range(0, 41):
                candidate_positions.add(d)
            
            candidate_positions = sorted(candidate_positions)
            
            for pos in candidate_positions:
                if pos in exponents:
                    continue  # Already have a term here
                
                for coeff in [1, -1, 2, -2, 3, -3, 4, -4]:
                    # Build candidate polynomial
                    new_exponents = exponents + [pos]
                    new_coeffs = coeffs + [coeff]
                    new_poly = make_poly_from_exponents(new_exponents, new_coeffs)
                    
                    if len(new_poly) > 41:
                        continue
                    
                    # Check it has exactly target_terms nonzero terms
                    nonzero = sum(1 for c in new_poly if c != 0)
                    if nonzero != target_terms:
                        continue
                    
                    # Compute square nonzero count
                    cnt = square_nonzero_count(new_poly)
                    
                    if cnt < best_score:
                        best_score = cnt
                        best_new_exp = pos
                        best_new_coeff = coeff
            
            if best_new_exp is not None:
                exponents.append(best_new_exp)
                coeffs.append(best_new_coeff)
            else:
                # Fallback: just add a term at max_deg + 1
                exponents.append(max(exponents) + 1)
                coeffs.append(1)
        
        poly = make_poly_from_exponents(exponents, coeffs)
        poly = trim(poly)
        return poly
    
    # Run greedy build
    if time.time() < deadline:
        greedy_poly = greedy_build()
        if is_valid(greedy_poly):
            cnt = square_nonzero_count(greedy_poly)
            if cnt < best_count:
                best_count = cnt
                best_poly = greedy_poly
    
    # Strategy 2: Simulated annealing on the space of 12-term polynomials
    # Represent state as (exponents, coeffs) where exponents are 12 distinct positions in [0,40]
    # and coeffs are values in [-64, 64] \ {0}
    
    def simulated_annealing(start_poly, max_iters=50000):
        """Simulated annealing to minimize square nonzero count."""
        # Initialize from start_poly
        nonzero_positions = [i for i, c in enumerate(start_poly) if c != 0]
        nonzero_coeffs = [start_poly[i] for i in nonzero_positions]
        
        current_exponents = nonzero_positions[:]
        current_coeffs = nonzero_coeffs[:]
        current_poly = start_poly[:]
        current_count = square_nonzero_count(current_poly)
        
        best_local_exponents = current_exponents[:]
        best_local_coeffs = current_coeffs[:]
        best_local_poly = current_poly[:]
        best_local_count = current_count
        
        # Annealing schedule
        T_start = 5.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / max(max_iters, 1))
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            T = T_start * (cooling_rate ** iteration)
            
            # Generate a neighbor
            # With some probability, move a term; with some probability, change a coefficient
            if np.random.random() < 0.6:
                # Move a term to a new position
                idx = np.random.randint(0, 12)
                old_exp = current_exponents[idx]
                
                # Try nearby positions
                new_exp = old_exp + np.random.randint(-5, 6)
                if new_exp < 0 or new_exp >= 41:
                    new_exp = np.random.randint(0, 41)
                
                # Check for collision
                if new_exp in current_exponents and current_exponents.index(new_exp) != idx:
                    continue
                
                current_exponents[idx] = new_exp
            else:
                # Change a coefficient
                idx = np.random.randint(0, 12)
                old_coeff = current_coeffs[idx]
                new_coeff = old_coeff + np.random.choice([-3, -2, -1, 1, 2, 3])
                if new_coeff == 0:
                    new_coeff = 1 if old_coeff > 0 else -1
                if abs(new_coeff) > 64:
                    new_coeff = np.clip(new_coeff, -64, 64)
                    if new_coeff == 0:
                        new_coeff = 1
                current_coeffs[idx] = new_coeff
            
            # Build new polynomial
            new_poly = make_poly_from_exponents(current_exponents, current_coeffs)
            new_poly = trim(new_poly)
            
            # Validate
            if not is_valid(new_poly):
                # Revert
                if np.random.random() < 0.6:
                    current_exponents[idx] = old_exp
                else:
                    current_coeffs[idx] = old_coeff
                continue
            
            new_count = square_nonzero_count(new_poly)
            
            # Acceptance criterion
            delta = new_count - current_count
            if delta <= 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current_count = new_count
                current_poly = new_poly[:]
                
                if new_count < best_local_count:
                    best_local_count = new_count
                    best_local_exponents = current_exponents[:]
                    best_local_coeffs = current_coeffs[:]
                    best_local_poly = new_poly[:]
            else:
                # Revert
                if np.random.random() < 0.6:
                    current_exponents[idx] = old_exp
                else:
                    current_coeffs[idx] = old_coeff
        
        return best_local_poly, best_local_count
    
    # Run simulated annealing from multiple starting points
    if time.time() < deadline:
        # Starting point 1: the current best
        sa_poly, sa_count = simulated_annealing(best_poly, max_iters=30000)
        if is_valid(sa_poly) and sa_count < best_count:
            best_count = sa_count
            best_poly = sa_poly
        
        # Starting point 2: greedy result
        if time.time() < deadline:
            sa_poly2, sa_count2 = simulated_annealing(greedy_poly, max_iters=30000)
            if is_valid(sa_poly2) and sa_count2 < best_count:
                best_count = sa_count2
                best_poly = sa_poly2
    
    # Strategy 3: Structured search with specific exponent patterns
    # Try patterns that are known to produce cancellations
    if time.time() < deadline:
        # Pattern: symmetric around a center
        # e.g., exponents at {c-d, c-d+1, ..., c+d} for some range
        # Or: pairs that sum to a constant
        
        # Try palindromic structures
        for center in range(10, 30):
            if time.time() > deadline:
                break
            
            # Generate symmetric exponent sets
            for spread in range(5, 15):
                if time.time() > deadline:
                    break
                
                # Symmetric exponents around center
                exps = []
                for d in range(spread):
                    e1 = center - d
                    e2 = center + d
                    if 0 <= e1 <= 40:
                        exps.append(e1)
                    if 0 <= e2 <= 40 and e2 != e1:
                        exps.append(e2)
                
                # Trim to 12
                if len(exps) > 12:
                    exps = exps[:12]
                if len(exps) < 12:
                    # Fill with nearby positions
                    for e in range(41):
                        if e not in exps and len(exps) < 12:
                            exps.append(e)
                
                if len(exps) != 12:
                    continue
                
                # Try sign patterns (fix first = +1)
                for signs in product([1, -1], repeat=11):
                    if time.time() > deadline:
                        break
                    
                    coeffs = [1] + list(signs)
                    poly = make_poly_from_exponents(exps, coeffs)
                    poly = trim(poly)
                    
                    if not is_valid(poly):
                        continue
                    
                    cnt = square_nonzero_count(poly)
                    if cnt < best_count:
                        best_count = cnt
                        best_poly = poly
    
    # Strategy 4: Local search with coefficient optimization
    if time.time() < deadline:
        # Take the best polynomial and try to optimize coefficients
        work_poly = best_poly[:]
        work_count = best_count
        
        for iteration in range(20000):
            if time.time() > deadline:
                break
            
            # Pick a random position to modify
            pos = np.random.randint(0, len(work_poly))
            old_val = work_poly[pos]
            
            # Try a new value
            new_val = old_val + np.random.choice([-3, -2, -1, 1, 2, 3])
            if new_val == 0 and old_val != 0:
                # This would reduce nonzero count, skip
                continue
            if new_val == 0:
                # This would add a nonzero, skip
                continue
            if abs(new_val) > 64:
                continue
            
            work_poly[pos] = new_val
            
            # Check validity
            nonzero = sum(1 for c in work_poly if c != 0)
            if nonzero != 12:
                work_poly[pos] = old_val
                continue
            
            if work_poly[-1] == 0:
                work_poly[pos] = old_val
                continue
            
            cnt = square_nonzero_count(work_poly)
            
            if cnt <= work_count:
                work_count = cnt
                if cnt < best_count:
                    best_count = cnt
                    best_poly = work_poly[:]
            else:
                work_poly[pos] = old_val
    
    # Final cleanup
    best_poly = trim(best_poly)
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    if not is_valid(best_poly):
        return [1] * 12
    
    return best_poly
