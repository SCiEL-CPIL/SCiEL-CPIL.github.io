import random
import time
import itertools
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 45.0  # safety margin before 90s wall-clock
    
    best_nonzero = 12
    best_poly = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 12 ones at even positions
    
    def make_poly(exponents, coeffs):
        """Create coefficient list from exponents and coefficients."""
        max_exp = max(exponents)
        poly = [0] * (max_exp + 1)
        for e, c in zip(exponents, coeffs):
            poly[e] = c
        return poly
    
    def square_nonzero(poly):
        """Count nonzero terms in the exact square of poly."""
        n = len(poly)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            if poly[i] != 0:
                for j in range(n):
                    if poly[j] != 0:
                        sq[i + j] += poly[i] * poly[j]
        return sum(1 for x in sq if x != 0)
    
    def valid(poly):
        """Check if poly meets all constraints."""
        if len(poly) > 41:
            return False
        if len(poly) == 0 or poly[-1] == 0:
            return False
        nonzero = [x for x in poly if x != 0]
        if len(nonzero) != 12:
            return False
        for x in nonzero:
            if not (isinstance(x, int) and -64 <= x <= 64):
                return False
        return True
    
    def try_poly(poly):
        nonlocal best_nonzero, best_poly
        if not valid(poly):
            return
        nz = square_nonzero(poly)
        if nz < best_nonzero:
            best_nonzero = nz
            best_poly = poly
    
    # Strategy 1: Try structured exponent sets with coefficients in {-1, 1}
    # Then refine to {-2, -1, 1, 2}
    
    # Try various exponent patterns
    base_exponents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # Try sign assignments on consecutive exponents
    for trial in range(2000):
        if time.time() > deadline:
            break
        signs = [random.choice([-1, 1]) for _ in range(12)]
        poly = make_poly(base_exponents, signs)
        try_poly(poly)
    
    # Try geometric-like exponents
    for base in [2, 3]:
        for trial in range(500):
            if time.time() > deadline:
                break
            exponents = [base ** i for i in range(12)]
            if max(exponents) > 40:
                exponents = list(range(12))
            signs = [random.choice([-1, 1]) for _ in range(12)]
            poly = make_poly(sorted(set(exponents)), signs)
            # Ensure exactly 12 terms
            if len(exponents) < 12:
                continue
            try_poly(poly)
    
    # Strategy 2: Local search - start from good candidates and perturb
    # First, do a more systematic search with different exponent sets
    
    # Try arithmetic progressions with various starts and steps
    for start_exp in range(0, 5):
        for step in [1, 2, 3]:
            exponents = [start_exp + step * i for i in range(12)]
            if max(exponents) > 40:
                continue
            # Try all sign patterns (2^12 = 4096)
            for mask in range(4096):
                if time.time() > deadline:
                    break
                signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
                poly = make_poly(exponents, signs)
                try_poly(poly)
    
    # Strategy 3: Refine coefficients to {-2, -1, 1, 2} from best found
    if best_poly is not None:
        # Local search on coefficients
        current = best_poly[:]
        current_nz = square_nonzero(current)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                old_val = current[i]
                for new_val in [-2, -1, 1, 2]:
                    if new_val == old_val:
                        continue
                    current[i] = new_val
                    if not valid(current):
                        current[i] = old_val
                        continue
                    nz = square_nonzero(current)
                    if nz < current_nz:
                        current_nz = nz
                        improved = True
                        if nz < best_nonzero:
                            best_nonzero = nz
                            best_poly = current[:]
                    else:
                        current[i] = old_val
    
    # Strategy 4: Random search with random exponents and small integer coefficients
    for trial in range(5000):
        if time.time() > deadline:
            break
        # Generate random exponent set
        exponents = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        poly = make_poly(exponents, coeffs)
        try_poly(poly)
    
    # Strategy 5: Greedy construction - build polynomial term by term
    # Start with a base term and add terms that maximize cancellation
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        # Start with a random first term
        first_exp = random.randint(0, 35)
        first_coeff = random.choice([-2, -1, 1, 2])
        exponents = [first_exp]
        coeffs = [first_coeff]
        
        for _ in range(11):
            # Try to add a term that reduces the square's nonzero count
            best_add_exp = None
            best_add_coeff = None
            best_nz = square_nonzero(make_poly(exponents, coeffs)) + 10  # high initial
            
            # Try candidate positions
            candidates = set()
            for e in exponents:
                for delta in range(-5, 6):
                    ne = e + delta
                    if 0 <= ne <= 40 and ne not in exponents:
                        candidates.add(ne)
            # Also try random positions
            for _ in range(10):
                candidates.add(random.randint(0, 40))
            
            for ne in sorted(candidates):
                if ne in exponents:
                    continue
                for nc in [-2, -1, 1, 2]:
                    test_exps = exponents + [ne]
                    test_coeffs = coeffs + [nc]
                    poly = make_poly(test_exps, test_coeffs)
                    nz = square_nonzero(poly)
                    if nz < best_nz:
                        best_nz = nz
                        best_add_exp = ne
                        best_add_coeff = nc
            
            if best_add_exp is not None:
                exponents.append(best_add_exp)
                coeffs.append(best_add_coeff)
            else:
                # Fallback: random addition
                ne = random.randint(0, 40)
                while ne in exponents:
                    ne = random.randint(0, 40)
                nc = random.choice([-2, -1, 1, 2])
                exponents.append(ne)
                coeffs.append(nc)
        
        exponents.sort()
        coeffs = [coeffs[i] for i in sorted(range(12), key=lambda x: exponents.__getitem__)]
        # Actually need to re-sort properly
        pairs = sorted(zip(exponents, coeffs))
        exponents = [p[0] for p in pairs]
        coeffs = [p[1] for p in pairs]
        poly = make_poly(exponents, coeffs)
        try_poly(poly)
    
    return best_poly
