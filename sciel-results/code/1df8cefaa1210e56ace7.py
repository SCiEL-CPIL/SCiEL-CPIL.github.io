"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def compute_square_nonzero(terms):
        """Given list of (exp, coeff), compute number of nonzero terms in square."""
        # terms: list of (exp, coeff)
        n = len(terms)
        # Use a dict to accumulate
        sq = {}
        for i in range(n):
            e1, c1 = terms[i]
            for j in range(i, n):
                e2, c2 = terms[j]
                exp = e1 + e2
                if i == j:
                    val = c1 * c1
                else:
                    val = 2 * c1 * c2
                if exp in sq:
                    sq[exp] += val
                else:
                    sq[exp] = val
        # Count nonzero
        return sum(1 for v in sq.values() if v != 0)
    
    def make_poly(exponents, coeffs):
        """Create polynomial as list in ascending degree."""
        max_exp = max(exponents)
        poly = [0] * (max_exp + 1)
        for e, c in zip(exponents, coeffs):
            poly[e] = c
        # Trim trailing zeros but keep length >= 1
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        return poly
    
    def poly_to_terms(poly):
        """Convert coefficient list to list of (exp, coeff) for nonzero terms."""
        return [(i, c) for i, c in enumerate(poly) if c != 0]
    
    def score_poly(terms):
        """Score = number of nonzero terms in square."""
        return compute_square_nonzero(terms)
    
    # Random initialization
    def random_poly():
        """Generate random 12-term polynomial."""
        # Choose 12 distinct exponents in [0, 40]
        exponents = random.sample(range(0, 41), 12)
        exponents.sort()
        # Choose coefficients in [-64, 64], nonzero
        coeffs = [random.choice([-64 + random.randint(0, 128)]) for _ in range(12)]
        # Ensure nonzero
        coeffs = [c if c != 0 else 1 for c in coeffs]
        poly = make_poly(exponents, coeffs)
        # Verify exactly 12 nonzero terms
        assert len(poly_to_terms(poly)) == 12
        return poly
    
    # Local search / simulated annealing
    def local_search(poly, max_iter=5000):
        terms = poly_to_terms(poly)
        best_score = score_poly(terms)
        best_poly = poly[:]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Random perturbation
            new_poly = poly[:]
            # Pick a random nonzero term and change it
            nz_indices = [i for i, c in enumerate(new_poly) if c != 0]
            if not nz_indices:
                break
            idx = random.choice(nz_indices)
            
            # Choose perturbation type
            ptype = random.random()
            if ptype < 0.3:
                # Change coefficient
                new_poly[idx] = random.randint(-64, 64)
                if new_poly[idx] == 0:
                    # Would reduce term count, so pick another
                    new_poly[idx] = random.choice([-1, 1, -2, 2, -64, 64])
            elif ptype < 0.5:
                # Move exponent: set current to 0, set a new position
                old_coeff = new_poly[idx]
                new_poly[idx] = 0
                # Pick a new exponent
                new_exp = random.randint(0, 40)
                if new_exp >= len(new_poly):
                    new_poly.extend([0] * (new_exp - len(new_poly) + 1))
                if new_poly[new_exp] == 0:
                    new_poly[new_exp] = old_coeff
                else:
                    # Collision, skip this move
                    new_poly[idx] = old_coeff
                    continue
            elif ptype < 0.7:
                # Swap exponents of two nonzero terms
                if len(nz_indices) >= 2:
                    i1, i2 = random.sample(nz_indices, 2)
                    new_poly[i1], new_poly[i2] = new_poly[i2], new_poly[i1]
            else:
                # Add a new term and remove one (keep 12 terms)
                # Remove a random nonzero
                remove_idx = random.choice(nz_indices)
                new_poly[remove_idx] = 0
                # Add at a new position
                new_exp = random.randint(0, 40)
                if new_exp >= len(new_poly):
                    new_poly.extend([0] * (new_exp - len(new_poly) + 1))
                if new_poly[new_exp] == 0:
                    new_poly[new_exp] = random.choice([-1, 1, -2, 2, -64, 64, random.randint(-64, 64)])
                    if new_poly[new_exp] == 0:
                        new_poly[new_exp] = 1
            
            # Validate: exactly 12 nonzero terms, degree <= 40, coeffs in [-64,64]
            new_terms = poly_to_terms(new_poly)
            if len(new_terms) != 12:
                continue
            if len(new_poly) > 41:
                continue
            if any(c < -64 or c > 64 for c in new_poly):
                continue
            if new_poly[-1] == 0:
                # Trim
                while len(new_poly) > 1 and new_poly[-1] == 0:
                    new_poly.pop()
            
            new_score = score_poly(new_terms)
            if new_score <= best_score:
                best_score = new_score
                best_poly = new_poly[:]
                poly = new_poly[:]
        
        return best_poly, best_score
    
    # Main search loop
    best_overall_score = float('inf')
    best_overall_poly = None
    
    # Try some structured initializations
    # 1. All ones at consecutive positions
    init1 = [0] * 12 + [0] * 29
    for i in range(12):
        init1[i] = 1
    # Actually make_poly handles this
    init1 = make_poly(list(range(12)), [1]*12)
    
    # 2. Alternating signs
    init2 = make_poly(list(range(12)), [1 if i%2==0 else -1 for i in range(12)])
    
    # 3. Powers of 2 exponents
    init3 = make_poly([0,1,2,4,8,16, 20,24,28,32,36,40], [1]*12)
    
    # 4. Symmetric
    init4 = make_poly([0,1,2,3,4,5, 35,36,37,38,39,40], [1]*12)
    
    candidates = [init1, init2, init3, init4]
    
    # Also add random ones
    for _ in range(5):
        if time.time() > deadline:
            break
        candidates.append(random_poly())
    
    for poly in candidates:
        if time.time() > deadline:
            break
        terms = poly_to_terms(poly)
        s = score_poly(terms)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_poly = poly[:]
    
    # Local search from each candidate
    for poly in candidates:
        if time.time() > deadline:
            break
        poly, s = local_search(poly, max_iter=3000)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_poly = poly[:]
    
    # More random restarts
    while time.time() < deadline - 2:
        poly = random_poly()
        poly, s = local_search(poly, max_iter=2000)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_poly = poly[:]
    
    if best_overall_poly is None:
        # Fallback
        best_overall_poly = [1] * 12
    
    # Final validation
    terms = poly_to_terms(best_overall_poly)
    assert len(terms) == 12, f"Expected 12 terms, got {len(terms)}"
    assert len(best_overall_poly) <= 41, f"Length {len(best_overall_poly)} > 41"
    assert best_overall_poly[-1] != 0, "Last coefficient is zero"
    assert all(-64 <= c <= 64 for c in best_overall_poly), "Coefficient out of range"
    
    return best_overall_poly
# EVOLVE-BLOCK-END

