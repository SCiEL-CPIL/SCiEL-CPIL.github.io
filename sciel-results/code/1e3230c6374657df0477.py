import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def square_nonzero_count(coeffs):
        """Compute square of polynomial given as coeff list, return count of nonzero terms."""
        n = len(coeffs)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(i, n):
                if coeffs[j] == 0:
                    continue
                val = coeffs[i] * coeffs[j]
                if i == j:
                    sq[i + j] += val
                else:
                    sq[i + j] += 2 * val
        return sum(1 for v in sq if v != 0)
    
    def is_valid(coeffs):
        """Check all constraints."""
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0 or coeffs[-1] == 0:
            return False
        # Trim trailing zeros for nonzero final coefficient check
        # Actually the spec says "nonzero final coefficient" meaning coeffs[-1] != 0
        if coeffs[-1] == 0:
            return False
        nz = sum(1 for c in coeffs if c != 0)
        if nz != 12:
            return False
        for c in coeffs:
            if c < -64 or c > 64:
                return False
        return True
    
    def count_nonzero(coeffs):
        return sum(1 for c in coeffs if c != 0)
    
    # Start with a valid candidate
    best_coeffs = None
    best_score = float('inf')
    
    # Initial: try to find a good starting point
    # Use factorization approach: P = A * B
    # Try small random polynomials
    
    def random_poly(n_terms, max_degree, max_coeff):
        """Generate random polynomial with exactly n_terms nonzero terms."""
        exponents = random.sample(range(max_degree + 1), n_terms)
        coeffs = [0] * (max_degree + 1)
        for e in exponents:
            coeffs[e] = random.randint(-max_coeff, max_coeff)
            while coeffs[e] == 0:
                coeffs[e] = random.randint(-max_coeff, max_coeff)
        return coeffs
    
    def local_search(coeffs, time_limit):
        """Improve a candidate via local search."""
        start = time.time()
        current = list(coeffs)
        current_score = square_nonzero_count(current)
        
        while time.time() - start < time_limit:
            improved = False
            # Try changing one coefficient
            for pos in range(len(current)):
                old_val = current[pos]
                for new_val in [-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64]:
                    if new_val == old_val:
                        continue
                    current[pos] = new_val
                    if not is_valid(current):
                        continue
                    score = square_nonzero_count(current)
                    if score < current_score:
                        current_score = score
                        improved = True
                        break
                    current[pos] = old_val
                if improved:
                    break
                current[pos] = old_val
            
            if not improved:
                # Try adding/removing terms
                # Remove a term
                for pos in range(len(current)):
                    if current[pos] != 0:
                        old_val = current[pos]
                        current[pos] = 0
                        if is_valid(current):
                            score = square_nonzero_count(current)
                            if score < current_score:
                                current_score = score
                                improved = True
                                break
                        current[pos] = old_val
                if improved:
                    continue
                    
                # Add a term
                for pos in range(len(current)):
                    if current[pos] == 0:
                        for new_val in [1, -1, 2, -2, 4, -4]:
                            current[pos] = new_val
                            if is_valid(current):
                                score = square_nonzero_count(current)
                                if score < current_score:
                                    current_score = score
                                    improved = True
                                    break
                        if improved:
                            break
                        current[pos] = 0
                if improved:
                    continue
                
                # Try expanding degree
                if len(current) < 41:
                    new_val = random.choice([1, -1, 2, -2])
                    current.append(new_val)
                    if is_valid(current):
                        score = square_nonzero_count(current)
                        if score < current_score:
                            current_score = score
                            improved = True
                        else:
                            current.pop()
                if improved:
                    continue
                
                # Try contracting degree (if trailing zero)
                if len(current) > 1 and current[-1] == 0:
                    current.pop()
                    if is_valid(current):
                        score = square_nonzero_count(current)
                        if score < current_score:
                            current_score = score
                            improved = True
                        else:
                            current.append(0)
                if improved:
                    continue
                    
                # No improvement, break
                break
        
        return current, current_score
    
    # Strategy 1: Factorization-based construction
    # P = A * B where A has a terms, B has b terms, a*b = 12
    # Factors: (1,12), (2,6), (3,4), (4,3), (6,2), (12,1)
    
    factor_pairs = [(2, 6), (3, 4), (4, 3), (6, 2)]
    
    for a_terms, b_terms in factor_pairs:
        if time.time() > deadline:
            break
        for attempt in range(500):
            if time.time() > deadline:
                break
            # Generate A and B
            max_deg = 20
            A = random_poly(a_terms, max_deg, 4)
            B = random_poly(b_terms, max_deg, 4)
            
            # Compute P = A * B
            P = [0] * (len(A) + len(B) - 1)
            for i, ai in enumerate(A):
                if ai == 0:
                    continue
                for j, bj in enumerate(B):
                    if bj == 0:
                        continue
                    P[i + j] += ai * bj
            
            # Trim trailing zeros
            while len(P) > 1 and P[-1] == 0:
                P.pop()
            
            if len(P) > 41:
                continue
            
            # Check if P has exactly 12 nonzero terms
            nz = count_nonzero(P)
            if nz != 12:
                continue
            
            # Check coefficient bounds
            valid = True
            for c in P:
                if c < -64 or c > 64:
                    valid = False
                    break
            if not valid:
                continue
            
            score = square_nonzero_count(P)
            if score < best_score:
                best_score = score
                best_coeffs = list(P)
            
            # Local search from this starting point
            if time.time() < deadline - 10:
                improved, imp_score = local_search(P, 2)
                if imp_score < best_score:
                    best_score = imp_score
                    best_coeffs = list(improved)
    
    # Strategy 2: Random search with local improvement
    if time.time() < deadline - 5:
        for attempt in range(2000):
            if time.time() > deadline - 5:
                break
            max_deg = random.randint(10, 40)
            P = random_poly(12, max_deg, 8)
            
            # Ensure nonzero final coefficient
            if P[-1] == 0:
                P[-1] = random.choice([1, -1, 2, -2])
            
            if len(P) > 41:
                continue
            
            valid = True
            for c in P:
                if c < -64 or c > 64:
                    valid = False
                    break
            if not valid:
                continue
            
            score = square_nonzero_count(P)
            if score < best_score:
                best_score = score
                best_coeffs = list(P)
            
            if time.time() < deadline - 5:
                improved, imp_score = local_search(P, 1)
                if imp_score < best_score:
                    best_score = imp_score
                    best_coeffs = list(improved)
    
    # Strategy 3: Try known sparse structures
    # (x^n - 1)^2 = x^{2n} - 2x^n + 1
    # Build P from such factors
    if time.time() < deadline - 3:
        for n in range(1, 20):
            if time.time() > deadline - 3:
                break
            # A = x^n - 1, B = some 6-term poly
            for _ in range(100):
                if time.time() > deadline - 3:
                    break
                B = random_poly(6, 20, 3)
                A = [0] * (n + 1)
                A[0] = -1
                A[n] = 1
                P = [0] * (len(A) + len(B) - 1)
                for i, ai in enumerate(A):
                    if ai == 0:
                        continue
                    for j, bj in enumerate(B):
                        if bj == 0:
                            continue
                        P[i + j] += ai * bj
                while len(P) > 1 and P[-1] == 0:
                    P.pop()
                if len(P) > 41:
                    continue
                if count_nonzero(P) != 12:
                    continue
                valid = all(-64 <= c <= 64 for c in P)
                if not valid:
                    continue
                score = square_nonzero_count(P)
                if score < best_score:
                    best_score = score
                    best_coeffs = list(P)
    
    # If we found something, return it; otherwise return a valid fallback
    if best_coeffs is not None and is_valid(best_coeffs):
        return best_coeffs
    
    # Fallback: return a valid 12-term polynomial
    fallback = [0] * 20
    exponents = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20]
    for i, e in enumerate(exponents):
        fallback[e] = 1
    # Ensure last coeff nonzero
    if fallback[-1] == 0:
        fallback[-1] = 1
    return fallback

