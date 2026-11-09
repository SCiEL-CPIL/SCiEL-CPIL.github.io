import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe
    
    best_poly = None
    best_score = float('inf')
    
    # Initial candidate: Seed
    seed_poly = [0]*12
    for i in range(12):
        seed_poly[i] = 1
    # Convert to dict or list of (pos, coeff)
    # We represent polynomial as a list of coefficients in ascending degree.
    # Max degree 40.
    
    def get_poly_from_terms(terms):
        # terms: list of (degree, coeff)
        max_deg = max([d for d, c in terms] + [0])
        poly = [0] * (max_deg + 1)
        for d, c in terms:
            if 0 <= d <= 40:
                poly[d] += c
        # Trim trailing zeros
        while poly and poly[-1] == 0:
            poly.pop()
        if not poly:
            poly = [0]
        return poly

    def score_poly(poly):
        if len(poly) > 41:
            return float('inf')
        if poly[-1] == 0:
            return float('inf')
        if any(abs(c) > 64 for c in poly):
            return float('inf')
        
        # Count non-zero terms in input
        nz_in = sum(1 for c in poly if c != 0)
        if nz_in != 12:
            return float('inf')
            
        # Calculate square
        sq = np.convolve(poly, poly)
        nz_sq = np.count_nonzero(sq)
        return nz_sq

    # Evaluate seed
    seed_score = score_poly(seed_poly)
    if seed_score < best_score:
        best_score = seed_score
        best_poly = seed_poly[:]

    # Local Search
    # Representation: list of 12 (degree, coeff) pairs
    # Degrees must be unique? No, if two terms have same degree, they combine.
    # So we should maintain a set of degrees or ensure uniqueness during moves.
    # Simpler: maintain a list of 12 unique degrees and 12 coeffs.
    
    def random_initial():
        # Random 12 distinct degrees in [0, 40]
        degs = random.sample(range(41), 12)
        coeffs = [random.choice([1, -1, 2, -2]) for _ in range(12)]
        return degs, coeffs

    def evaluate_terms(degs, coeffs):
        # Combine if any degs are same (shouldn't happen if we manage carefully)
        # But for safety, build poly
        max_deg = max(degs)
        poly = [0] * (max_deg + 1)
        for d, c in zip(degs, coeffs):
            poly[d] = c
        return score_poly(poly)

    def move(degs, coeffs):
        new_degs = degs[:]
        new_coeffs = coeffs[:]
        move_type = random.choice(['flip', 'shift', 'mag'])
        idx = random.randint(0, 11)
        
        if move_type == 'flip':
            new_coeffs[idx] *= -1
        elif move_type == 'shift':
            shift = random.choice([-1, 1])
            new_deg = new_degs[idx] + shift
            if 0 <= new_deg <= 40 and new_deg not in new_degs:
                new_degs[idx] = new_deg
            else:
                # Try another index or skip
                return degs, coeffs
        else: # mag
            new_coeffs[idx] = random.choice([1, -1, 2, -2, 3, -3])
            
        return new_degs, new_coeffs

    # Run search
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Initialize
        degs, coeffs = random_initial()
        curr_score = evaluate_terms(degs, coeffs)
        
        # Simulated Annealing / Hill Climbing
        for step in range(1000):
            if time.time() > deadline:
                break
                
            if curr_score < best_score:
                best_score = curr_score
                # Reconstruct poly for best
                max_deg = max(degs)
                best_poly = [0] * (max_deg + 1)
                for d, c in zip(degs, coeffs):
                    best_poly[d] = c
                # Ensure format
                while best_poly and best_poly[-1] == 0:
                    best_poly.pop()
                if not best_poly: best_poly = [0]
                
            new_degs, new_coeffs = move(degs, coeffs)
            new_score = evaluate_terms(new_degs, new_coeffs)
            
            # Accept if better or random (annealing)
            if new_score < curr_score or random.random() < 0.05:
                degs, coeffs = new_degs, new_coeffs
                curr_score = new_score
                
            if curr_score <= 5: # Found a very good one
                break

    if best_poly is None:
        return seed_poly
        
    return best_poly

construct()
