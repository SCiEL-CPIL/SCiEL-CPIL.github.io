import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill
    
    best_score = float('inf')
    best_poly = None
    
    # Helper to count nonzero terms in square
    def count_nz_square(poly):
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        return int(np.count_nonzero(sq))
    
    # Helper to validate polynomial
    def is_valid(poly):
        if len(poly) != 41:
            return False
        if poly[40] == 0:
            return False
        nz_count = sum(1 for x in poly if x != 0)
        if nz_count != 12:
            return False
        if any(abs(x) > 64 for x in poly):
            return False
        return True
    
    # Strategy: Use decomposition P = A + B where A and B are sparse
    # Key idea: If B = -A shifted, then P = A - A(t^k)
    # P^2 = A^2 - 2*A(t)*A(t^k) + A^2(t^k)
    # The cross term can cancel many terms
    
    # Generate candidate base polynomials A with 6-7 terms
    # Then construct P = A + B where B is derived from A
    
    def generate_base_poly():
        """Generate a sparse base polynomial with 6-7 terms"""
        num_terms = random.choice([6, 7])
        indices = random.sample(range(41), num_terms)
        # Ensure we can shift and still stay in range
        # Prefer indices that allow shifting
        signs = [random.choice([-1, 1]) for _ in range(num_terms)]
        poly = [0] * 41
        for idx, sgn in zip(indices, signs):
            poly[idx] = sgn
        return poly, indices, signs
    
    def construct_from_base(A, shift, coeff):
        """
        Construct P = A + coeff * A(t^shift)
        Returns the polynomial if valid, else None
        """
        B = [0] * 41
        for i, val in enumerate(A):
            if val != 0:
                new_idx = i + shift
                if 0 <= new_idx < 41:
                    B[new_idx] = val * coeff
        
        P = [a + b for a, b in zip(A, B)]
        
        # Check if P has exactly 12 nonzero terms
        nz_count = sum(1 for x in P if x != 0)
        if nz_count != 12:
            return None
        
        # Check degree constraint: P[40] must be nonzero
        if P[40] == 0:
            return None
        
        # Check coefficient bounds
        if any(abs(x) > 64 for x in P):
            return None
        
        return P
    
    # Try various constructions
    # Type 1: P = A - A(t^k) (coeff = -1)
    # Type 2: P = A + A(t^k) (coeff = 1)
    # Type 3: P = 2*A - A(t^k) (coeff = -1, but A scaled by 2)
    # Type 4: P = A - 2*A(t^k) (coeff = -2)
    
    attempts = 0
    max_attempts = 50000
    
    while attempts < max_attempts and time.time() < deadline - 5.0:
        attempts += 1
        
        # Generate base polynomial
        A, indices, signs = generate_base_poly()
        
        # Try different shifts and coefficients
        for shift in range(1, 15):
            if time.time() > deadline - 5.0:
                break
                
            for coeff in [-2, -1, 1, 2]:
                P = construct_from_base(A, shift, coeff)
                if P is None:
                    continue
                    
                score = count_nz_square(P)
                if score < best_score:
                    best_score = score
                    best_poly = P[:]
        
        # Also try: P = A + B where B is a different sparse polynomial
        # This is more general but harder to search
        if attempts % 10 == 0:
            # Generate two independent sparse polynomials
            A1, _, _ = generate_base_poly()
            A2, _, _ = generate_base_poly()
            
            # Try adding them with various coefficients
            for c1 in [-2, -1, 1, 2]:
                for c2 in [-2, -1, 1, 2]:
                    P = [a1 * c1 + a2 * c2 for a1, a2 in zip(A1, A2)]
                    
                    nz_count = sum(1 for x in P if x != 0)
                    if nz_count != 12:
                        continue
                    if P[40] == 0:
                        continue
                    if any(abs(x) > 64 for x in P):
                        continue
                    
                    score = count_nz_square(P)
                    if score < best_score:
                        best_score = score
                        best_poly = P[:]
    
    # If we found a good one, try to improve it with local search
    if best_poly is not None:
        current_poly = best_poly[:]
        current_score = best_score
        
        # Local search: try moving terms or changing values
        no_improve = 0
        while time.time() < deadline - 1.0 and no_improve < 500:
            neighbor = current_poly[:]
            move_type = random.random()
            
            if move_type < 0.5:
                # Change value
                nz_indices = [i for i in range(41) if neighbor[i] != 0]
                idx = random.choice(nz_indices)
                old_val = neighbor[idx]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                if new_val == 0:
                    continue
                if abs(new_val) > 64:
                    continue
                neighbor[idx] = new_val
            else:
                # Move term
                nz_indices = [i for i in range(41) if neighbor[i] != 0]
                z_indices = [i for i in range(41) if neighbor[i] == 0]
                if not z_indices:
                    continue
                src = random.choice(nz_indices)
                if src == 40:
                    continue
                dst = random.choice(z_indices)
                val = neighbor[src]
                neighbor[src] = 0
                neighbor[dst] = val
            
            if not is_valid(neighbor):
                continue
                
            new_score = count_nz_square(neighbor)
            
            if new_score < current_score:
                current_poly = neighbor
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                no_improve += 1
    
    # Fallback: if no valid poly found, create one
    if best_poly is None:
        # Create a valid polynomial with 12 terms
        best_poly = [0] * 41
        indices = random.sample(range(41), 12)
        if 40 not in indices:
            indices[-1] = 40
        for idx in indices:
            best_poly[idx] = random.choice([-1, 1])
        best_score = count_nz_square(best_poly)
    
    return best_poly
