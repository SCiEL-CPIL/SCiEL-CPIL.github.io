import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safe margin for 60s limit
    
    best_score = float('inf')
    best_poly = None
    
    def count_nz_square(poly):
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        return np.count_nonzero(sq)

    def construct_poly_from_B_D(B_coeffs, B_exponents, a0, a1, D):
        """
        Constructs P(x) = a0 * B(x) + a1 * x^D * B(x)
        B(x) is defined by B_coeffs and B_exponents.
        Returns the coefficient list of P if it has exactly 12 nonzero terms, else None.
        """
        p = [0] * 41
        # Add a0 * B(x)
        for c, e in zip(B_coeffs, B_exponents):
            if 0 <= e < 41:
                p[e] += a0 * c
            else:
                return None # Out of bounds
        
        # Add a1 * x^D * B(x)
        for c, e in zip(B_coeffs, B_exponents):
            new_e = e + D
            if 0 <= new_e < 41:
                p[new_e] += a1 * c
            else:
                return None # Out of bounds
        
        # Check exactly 12 nonzero terms
        nz_count = sum(1 for x in p if x != 0)
        if nz_count != 12:
            return None
        
        # Check bounds
        if any(abs(x) > 64 for x in p):
            return None
            
        return p

    def random_B():
        # Generate random 6-term polynomial B
        exponents = sorted(random.sample(range(41), 6))
        coeffs = [random.choice([-1, 1, -2, 2, -3, 3]) for _ in range(6)]
        return exponents, coeffs

    # Initial candidates
    candidates = []
    
    # Try some structured B's
    # 1. Consecutive
    exps = list(range(6))
    coefs = [1, -1, 1, -1, 1, -1]
    candidates.append((exps, coefs))
    
    # 2. Powers of 2 (sparse square)
    exps = [0, 1, 2, 4, 8, 16]
    coefs = [1, 1, 1, 1, 1, 1]
    candidates.append((exps, coefs))
    
    # 3. Random
    for _ in range(50):
        candidates.append(random_B())

    # Evaluate initial candidates
    for exps, coefs in candidates:
        # Try small shifts D and small a0, a1
        for D in range(1, 41):
            for a0 in [-1, 1, -2, 2]:
                for a1 in [-1, 1, -2, 2]:
                    p = construct_poly_from_B_D(coefs, exps, a0, a1, D)
                    if p is not None:
                        score = count_nz_square(p)
                        if score < best_score:
                            best_score = score
                            best_poly = p[:]

    if best_poly is None:
        # Fallback to random 12-term poly if structural search fails
        p = [0] * 41
        indices = random.sample(range(41), 12)
        for idx in indices:
            p[idx] = random.choice([-1, 1])
        if p[40] == 0:
            nz_indices = [i for i in range(41) if p[i] != 0]
            k = random.choice(nz_indices)
            p[40] = p[k]
            p[k] = 0
        best_poly = p
        best_score = count_nz_square(p)

    # Local Search on B and D
    current_exps, current_coefs = candidates[0]
    current_D = 1
    current_a0 = 1
    current_a1 = 1
    current_p = construct_poly_from_B_D(current_coefs, current_exps, current_a0, current_a1, current_D)
    if current_p is None:
        # Find a valid start
        for exps, coefs in candidates:
            for D in range(1, 41):
                p = construct_poly_from_B_D(coefs, exps, 1, 1, D)
                if p is not None:
                    current_exps, current_coefs = exps, coefs
                    current_D = D
                    current_a0, current_a1 = 1, 1
                    current_p = p
                    break
            if current_p is not None:
                break
    
    if current_p is None:
         return best_poly

    current_score = count_nz_square(current_p)
    if current_score < best_score:
        best_score = current_score
        best_poly = current_p[:]

    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        neighbor_exps = current_exps[:]
        neighbor_coefs = current_coefs[:]
        neighbor_D = current_D
        neighbor_a0 = current_a0
        neighbor_a1 = current_a1
        
        move_type = random.random()
        
        if move_type < 0.3:
            # Perturb exponents
            idx = random.randint(0, 5)
            delta = random.choice([-2, -1, 1, 2])
            new_exp = neighbor_exps[idx] + delta
            if 0 <= new_exp < 41 and new_exp not in neighbor_exps:
                neighbor_exps[idx] = new_exp
                neighbor_exps.sort()
            else:
                continue # Invalid move, skip
        elif move_type < 0.6:
            # Perturb coefficients
            idx = random.randint(0, 5)
            old_val = neighbor_coefs[idx]
            new_val = old_val + random.choice([-2, -1, 1, 2])
            if new_val != 0 and abs(new_val) <= 64:
                neighbor_coefs[idx] = new_val
            else:
                continue
        elif move_type < 0.8:
            # Perturb D
            delta_D = random.choice([-2, -1, 1, 2])
            new_D = neighbor_D + delta_D
            if new_D > 0:
                neighbor_D = new_D
            else:
                continue
        else:
            # Perturb a0, a1
            if random.random() < 0.5:
                neighbor_a0 = random.choice([-1, 1, -2, 2])
            else:
                neighbor_a1 = random.choice([-1, 1, -2, 2])

        neighbor_p = construct_poly_from_B_D(neighbor_coefs, neighbor_exps, neighbor_a0, neighbor_a1, neighbor_D)
        
        if neighbor_p is None:
            continue
            
        neighbor_score = count_nz_square(neighbor_p)
        
        # Acceptance
        if neighbor_score < current_score:
            current_exps, current_coefs = neighbor_exps, neighbor_coefs
            current_D = neighbor_D
            current_a0, current_a1 = neighbor_a0, neighbor_a1
            current_p = neighbor_p
            current_score = neighbor_score
            T = 1.0
            if current_score < best_score:
                best_score = current_score
                best_poly = current_p[:]
        else:
            delta = neighbor_score - current_score
            if T > T_min:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_exps, current_coefs = neighbor_exps, neighbor_coefs
                    current_D = neighbor_D
                    current_a0, current_a1 = neighbor_a0, neighbor_a1
                    current_p = neighbor_p
                    current_score = neighbor_score
            T *= cooling

    return best_poly
