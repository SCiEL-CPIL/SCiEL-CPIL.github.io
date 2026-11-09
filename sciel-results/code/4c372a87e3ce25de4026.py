import numpy as np
import time
import random
from itertools import combinations, product

def _sq_nonzero_count(coeffs):
    """Compute the number of nonzero terms in the square of a polynomial given by coefficient list."""
    sq = np.convolve(coeffs, coeffs)
    return int(np.count_nonzero(sq))

def _valid(coeffs):
    """Check if coeffs is a valid candidate: length<=41, nonzero final coeff, entries in [-64,64], exactly 12 nonzero terms."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    if any(abs(c) > 64 for c in coeffs):
        return False
    nz = sum(1 for c in coeffs if c != 0)
    if nz != 12:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safe margin for 90s limit
    
    best_score = float('inf')
    best_coeffs = None
    
    np.random.seed(42)
    random.seed(42)
    
    def count_nz_square(poly):
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        return int(np.count_nonzero(sq))
    
    def random_poly():
        p = [0] * 41
        indices = random.sample(range(41), 12)
        for idx in indices:
            p[idx] = random.choice([-1, 1])
        if p[40] == 0:
            nz_indices = [i for i in range(41) if p[i] != 0]
            k = random.choice(nz_indices)
            p[40] = p[k]
            p[k] = 0
        return p
    
    # Phase 1: Structured candidates based on algebraic decomposition
    # P = A + B where A, B are sparse, targeting cancellation in (A+B)^2 = A^2 + B^2 + 2AB
    
    candidates = []
    
    # 1a: Product of (1 - x^a) factors - creates exact cancellations
    # (1-x^a)(1-x^b)(1-x^c) = 1 - x^a - x^b - x^c + x^(a+b) + x^(a+c) + x^(b+c) - x^(a+b+c)
    # That's 8 terms. Multiply by (1-x^d) to get up to 16 terms, but overlaps reduce count.
    
    for a in range(1, 10):
        for b in range(a+1, 15):
            for c in range(b+1, 20):
                if time.time() > deadline:
                    break
                # (1-x^a)(1-x^b)(1-x^c) has 8 terms
                # Multiply by x^k * (1-x^d) to shift and add one more factor
                # Total terms: up to 16, but we need exactly 12
                
                # Try: (1-x^a)(1-x^b)(1-x^c) * x^shift
                # This gives 8 terms at exponents: 0, a, b, c, a+b, a+c, b+c, a+b+c (shifted)
                # That's only 8 terms. We need 12.
                
                # Alternative: (1-x^a)(1-x^b)(1-x^c)(1-x^d)
                # Terms: 1, -x^a, -x^b, -x^c, -x^d, +x^(a+b), +x^(a+c), +x^(a+d), +x^(b+c), +x^(b+d), +x^(c+d), -x^(a+b+c), -x^(a+b+d), -x^(a+c+d), -x^(b+c+d), +x^(a+b+c+d)
                # That's 16 terms max, but if some exponents collide, fewer.
                
                for d in range(c+1, 25):
                    if time.time() > deadline:
                        break
                    if a+b+c+d > 40:
                        continue
                    
                    # Build the polynomial
                    p = [0] * 41
                    # All subsets of {a,b,c,d}
                    for mask in range(16):
                        exp = 0
                        sign = 1
                        for j, x in enumerate([a, b, c, d]):
                            if mask & (1 << j):
                                exp += x
                                sign *= -1
                        if 0 <= exp < 41:
                            p[exp] += sign
                    
                    nz = sum(1 for x in p if x != 0)
                    if nz == 12 and p[40] != 0:
                        # Check if this is new
                        candidates.append(p[:])
                    elif nz == 12:
                        # Fix leading coefficient
                        nz_indices = [i for i in range(41) if p[i] != 0]
                        if nz_indices:
                            k = nz_indices[-1]
                            p[40] = p[k]
                            p[k] = 0
                            if sum(1 for x in p if x != 0) == 12:
                                candidates.append(p[:])
    
    # 1b: Arithmetic progression with alternating signs
    for start in range(-29, 1):
        if time.time() > deadline:
            break
        for step in range(1, 5):
            p = [0] * 41
            for i in range(12):
                idx = start + i * step
                if 0 <= idx < 41:
                    p[idx] = 1 if i % 2 == 0 else -1
            
            nz = sum(1 for x in p if x != 0)
            if nz == 12:
                if p[40] == 0:
                    nz_indices = [i for i in range(41) if p[i] != 0]
                    if nz_indices:
                        k = nz_indices[-1]
                        p[40] = p[k]
                        p[k] = 0
                        if sum(1 for x in p if x != 0) == 12:
                            candidates.append(p[:])
                else:
                    candidates.append(p[:])
    
    # 1c: Random sparse polynomials
    for _ in range(50):
        if time.time() > deadline:
            break
        candidates.append(random_poly())
    
    # 1d: Palindromic structures
    for d in range(12, 41, 2):
        if time.time() > deadline:
            break
        half = d // 2
        candidates_list = list(range(0, half + 1))
        for _ in range(30):
            if time.time() > deadline:
                break
            if len(candidates_list) < 6:
                continue
            chosen = random.sample(candidates_list, 6)
            exponents = []
            for c in chosen:
                if c == d - c:
                    exponents.append(c)
                else:
                    exponents.append(c)
                    exponents.append(d - c)
            exponents = sorted(set(exponents))
            if len(exponents) != 12:
                continue
            
            p = [0] * 41
            for i, e in enumerate(exponents):
                p[e] = random.choice([-1, 1])
            if p[40] == 0:
                nz_indices = [i for i in range(41) if p[i] != 0]
                if nz_indices:
                    k = nz_indices[-1]
                    p[40] = p[k]
                    p[k] = 0
            if sum(1 for x in p if x != 0) == 12:
                candidates.append(p[:])
    
    # Evaluate all candidates
    for p in candidates:
        if not _valid(p):
            continue
        score = count_nz_square(p)
        if score < best_score:
            best_score = score
            best_coeffs = p[:]
    
    if best_coeffs is None:
        best_coeffs = [0] * 41
        for i in range(12):
            best_coeffs[i] = 1
        best_score = count_nz_square(best_coeffs)
    
    # Phase 2: Simulated Annealing refinement
    current_poly = best_coeffs[:]
    current_score = best_score
    T = 1.0
    T_min = 0.01
    cooling = 0.9998
    no_improve = 0
    
    while time.time() < deadline:
        if no_improve > 500:
            # Restart with a new random candidate
            current_poly = random_poly()
            current_score = count_nz_square(current_poly)
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_poly[:]
            no_improve = 0
            T = 1.0
            continue
        
        # Generate a neighbor
        neighbor = current_poly[:]
        move_type = random.random()
        
        if move_type < 0.5:
            # Change coefficient value of a random non-zero term
            nz_indices = [i for i in range(41) if neighbor[i] != 0]
            idx = random.choice(nz_indices)
            old_val = neighbor[idx]
            new_val = old_val + random.choice([-2, -1, 1, 2])
            if new_val == 0:
                new_val = old_val
            elif abs(new_val) > 64:
                new_val = 64 if new_val > 0 else -64
            else:
                if random.random() < 0.15:
                    new_val = random.randint(-10, 10)
                    if new_val == 0:
                        new_val = 1
            neighbor[idx] = new_val
        elif move_type < 0.8:
            # Move a term to a new position
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
        else:
            # Swap two terms
            nz_indices = [i for i in range(41) if neighbor[i] != 0]
            if len(nz_indices) < 2:
                continue
            i1, i2 = random.sample(nz_indices, 2)
            if i1 == 40 or i2 == 40:
                continue
            neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
        
        if not _valid(neighbor):
            no_improve += 1
            T *= cooling
            continue
        
        new_score = count_nz_square(neighbor)
        
        if new_score < current_score:
            current_poly = neighbor
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_poly[:]
        else:
            delta = new_score - current_score
            if T > T_min:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_poly = neighbor
                    current_score = new_score
            no_improve += 1
            T *= cooling
    
    # Final validation
    if not _valid(best_coeffs):
        best_coeffs = [0] * 41
        for i in range(12):
            best_coeffs[i] = 1
    
    return best_coeffs
