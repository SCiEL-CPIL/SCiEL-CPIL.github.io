import time
import random
import itertools
from collections import defaultdict

def _square_nonzero_count(coeffs):
    """Compute number of nonzero terms in the square of a polynomial."""
    n = len(coeffs)
    # Square has degree 2*(n-1), so length 2*n-1
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(n):
            if coeffs[j] == 0:
                continue
            sq[i + j] += coeffs[i] * coeffs[j]
    return sum(1 for x in sq if x != 0)

def _square_coeff_list(coeffs):
    """Return the coefficient list of the square."""
    n = len(coeffs)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(n):
            if coeffs[j] == 0:
                continue
            sq[i + j] += coeffs[i] * coeffs[j]
    return sq

def _is_valid(coeffs):
    """Check if polynomial meets all constraints."""
    if len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    nonzero = sum(1 for c in coeffs if c != 0)
    if nonzero != 12:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    return True

def _trim_trailing_zeros(coeffs):
    """Remove trailing zeros while keeping at least one element."""
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs

def _try_geometric(start_exp, ratio, n_terms=12, coeff_scale=1):
    """Try a geometric progression of exponents with integer coefficients."""
    exponents = []
    for i in range(n_terms):
        exp = start_exp + i * ratio
        if exp > 40:
            break
        exponents.append(exp)
    if len(exponents) < n_terms:
        return None
    coeffs = [0] * (max(exponents) + 1)
    for i, exp in enumerate(exponents):
        coeffs[exp] = coeff_scale
    return coeffs

def _try_palindromic(base_coeffs):
    """Make a palindromic polynomial from a base set of coefficients."""
    n = len(base_coeffs)
    coeffs = [0] * (2 * n - 1)
    for i in range(n):
        coeffs[i] = base_coeffs[i]
        coeffs[2 * n - 2 - i] = base_coeffs[i]
    return coeffs

def _local_search(coeffs, max_iters=2000):
    """Local search: mutate exponents and coefficients, keep improvements."""
    best_coeffs = list(coeffs)
    best_score = _square_nonzero_count(best_coeffs)
    
    for _ in range(max_iters):
        # Random mutation
        new_coeffs = list(best_coeffs)
        mutation_type = random.choice(['coeff', 'exponent', 'add_remove'])
        
        if mutation_type == 'coeff':
            # Change a coefficient
            idx = random.randint(0, len(new_coeffs) - 1)
            new_val = random.randint(-64, 64)
            new_coeffs[idx] = new_val
        elif mutation_type == 'exponent':
            # Move a term to a different exponent
            nonzero_indices = [i for i, c in enumerate(new_coeffs) if c != 0]
            if nonzero_indices:
                src = random.choice(nonzero_indices)
                dst = random.randint(0, 40)
                val = new_coeffs[src]
                new_coeffs[src] = 0
                # Extend if needed
                if dst >= len(new_coeffs):
                    new_coeffs.extend([0] * (dst - len(new_coeffs) + 1))
                new_coeffs[dst] = val
        else:
            # Add or remove a term
            if random.random() < 0.5 and len([c for c in new_coeffs if c != 0]) < 12:
                # Add a term
                idx = random.randint(0, 40)
                if idx >= len(new_coeffs):
                    new_coeffs.extend([0] * (idx - len(new_coeffs) + 1))
                new_coeffs[idx] = random.randint(-64, 64)
            elif len([c for c in new_coeffs if c != 0]) > 12:
                # Remove a term
                nonzero_indices = [i for i, c in enumerate(new_coeffs) if c != 0]
                if nonzero_indices:
                    idx = random.choice(nonzero_indices)
                    new_coeffs[idx] = 0
        
        # Validate
        if not _is_valid(new_coeffs):
            continue
        
        score = _square_nonzero_count(new_coeffs)
        if score < best_score:
            best_coeffs = new_coeffs
            best_score = score
    
    return best_coeffs, best_score

def _structured_search():
    """Try structured families first."""
    best_coeffs = None
    best_score = float('inf')
    
    # Try geometric progressions
    for start in range(0, 10):
        for ratio in range(1, 5):
            for scale in [1, -1, 2, -2, 3]:
                coeffs = _try_geometric(start, ratio, 12, scale)
                if coeffs and _is_valid(coeffs):
                    score = _square_nonzero_count(coeffs)
                    if score < best_score:
                        best_score = score
                        best_coeffs = list(coeffs)
    
    # Try palindromic structures
    for _ in range(200):
        base = [random.choice([-1, 0, 1, 2, -2, 3, -3]) for _ in range(6)]
        coeffs = _try_palindromic(base)
        if _is_valid(coeffs):
            score = _square_nonzero_count(coeffs)
            if score < best_score:
                best_score = score
                best_coeffs = list(coeffs)
    
    return best_coeffs, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    random.seed(42)
    
    best_coeffs = None
    best_score = float('inf')
    
    # Phase 1: Structured search
    sc_coeffs, sc_score = _structured_search()
    if sc_coeffs:
        best_coeffs = sc_coeffs
        best_score = sc_score
    
    # Phase 2: Local search with random restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Generate random starting point
        n_terms = 12
        exponents = random.sample(range(0, 41), n_terms)
        exponents.sort()
        coeffs = [0] * 41
        for exp in exponents:
            coeffs[exp] = random.choice([-1, 1, -2, 2, -3, 3, 1, -1])
        
        # Ensure last coefficient is nonzero
        if coeffs[-1] == 0:
            coeffs[-1] = random.choice([-1, 1, 2])
        
        # Ensure exactly 12 nonzero terms
        nonzero = [i for i, c in enumerate(coeffs) if c != 0]
        if len(nonzero) != 12:
            # Adjust
            if len(nonzero) > 12:
                for i in nonzero[12:]:
                    coeffs[i] = 0
            elif len(nonzero) < 12:
                zeros = [i for i in range(41) if coeffs[i] == 0]
                random.shuffle(zeros)
                for i in zeros[:12 - len(nonzero)]:
                    coeffs[i] = random.choice([-1, 1, 2])
        
        if not _is_valid(coeffs):
            continue
        
        # Local search
        lc_coeffs, lc_score = _local_search(coeffs, max_iters=500)
        if lc_score < best_score:
            best_score = lc_score
            best_coeffs = lc_coeffs
        
        # Early termination if we find a very good solution
        if best_score <= 12:
            break
    
    # Return the best found, or a valid default
    if best_coeffs is None or not _is_valid(best_coeffs):
        # Fallback: simple valid polynomial
        best_coeffs = [0] * 12
        best_coeffs[0] = 1
        best_coeffs[11] = 1
        for i in range(2, 11):
            best_coeffs[i] = 1
    
    # Final validation and trim
    _trim_trailing_zeros(best_coeffs)
    
    return best_coeffs
