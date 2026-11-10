import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    def count_nonzero_square(coeffs, degrees):
        """Count nonzero terms in the square of a polynomial given by (coeffs, degrees)."""
        # Build full coefficient array
        max_deg = 2 * max(degrees)
        sq = [0] * (max_deg + 1)
        n = len(coeffs)
        for i in range(n):
            for j in range(i, n):
                deg = degrees[i] + degrees[j]
                if i == j:
                    sq[deg] += coeffs[i] * coeffs[j]
                else:
                    sq[deg] += 2 * coeffs[i] * coeffs[j]
        return sum(1 for c in sq if c != 0)
    
    def make_poly(degrees, coeffs):
        """Create coefficient list in ascending degree format."""
        max_deg = max(degrees) if degrees else 0
        result = [0] * (max_deg + 1)
        for d, c in zip(degrees, coeffs):
            result[d] = c
        return result
    
    best_score = float('inf')
    best_result = None
    
    # Initial structured candidates
    # 1. Arithmetic progression degrees with same coefficient
    for base_coeff in [1, -1, 2, -2]:
        for start in range(0, 10):
            for gap in [1, 2, 3]:
                degrees = [start + i * gap for i in range(12)]
                if max(degrees) > 40:
                    continue
                coeffs = [base_coeff] * 12
                score = count_nonzero_square(coeffs, degrees)
                if score < best_score:
                    best_score = score
                    best_result = make_poly(degrees, coeffs)
    
    # 2. Try some known good patterns
    # Palindromic structures
    for gap in [1, 2]:
        for center in range(0, 20):
            degrees = sorted(set([center + i * gap for i in range(-5, 6)]))
            if len(degrees) < 12:
                degrees = [center + i * gap for i in range(12)]
            if max(degrees) > 40:
                continue
            coeffs = [1] * 12
            score = count_nonzero_square(coeffs, degrees)
            if score < best_score:
                best_score = score
                best_result = make_poly(degrees, coeffs)
    
    # Local search
    def random_initial():
        degrees = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1, -2, 2, -3, 3, -4, 4]) for _ in range(12)]
        # Ensure nonzero
        return degrees, coeffs
    
    def local_search(degrees, coeffs, max_iters=5000):
        """Local search to minimize nonzero terms in square."""
        current_score = count_nonzero_square(coeffs, degrees)
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Pick a random move
            move_type = random.random()
            if move_type < 0.4:
                # Change a coefficient
                idx = random.randint(0, 11)
                old_c = coeffs[idx]
                new_c = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -8, 8, -16, 16, -32, 32, -64, 64])
                if new_c == old_c or new_c == 0:
                    continue
                coeffs[idx] = new_c
                new_score = count_nonzero_square(coeffs, degrees)
                if new_score <= current_score:
                    current_score = new_score
                else:
                    # Revert with some probability (simulated annealing-like)
                    if random.random() > 0.1:
                        coeffs[idx] = old_c
                    else:
                        current_score = new_score
            elif move_type < 0.7:
                # Move a degree
                idx = random.randint(0, 11)
                old_d = degrees[idx]
                new_d = random.randint(0, 40)
                if new_d == old_d:
                    continue
                # Check no collision
                if any(d == new_d for i, d in enumerate(degrees) if i != idx):
                    continue
                degrees[idx] = new_d
                degrees.sort()
                # Re-index coeffs to match sorted degrees
                # This is tricky, let me just rebuild
                # Actually, let me keep them paired
                # Redo: keep list of (degree, coeff) pairs
                new_score = count_nonzero_square(coeffs, degrees)
                if new_score <= current_score:
                    current_score = new_score
                else:
                    if random.random() > 0.1:
                        degrees[idx] = old_d
                        degrees.sort()
                    else:
                        current_score = new_score
            else:
                # Swap two degrees
                i, j = random.sample(range(12), 2)
                degrees[i], degrees[j] = degrees[j], degrees[i]
                degrees.sort()
                # Re-pair coeffs
                # This approach is getting messy. Let me use a different representation.
                new_score = count_nonzero_square(coeffs, degrees)
                if new_score <= current_score:
                    current_score = new_score
                else:
                    degrees[i], degrees[j] = degrees[j], degrees[i]
                    degrees.sort()
                    current_score = count_nonzero_square(coeffs, degrees)
        
        return current_score
    
    # Better representation: list of (degree, coeff) pairs
    def count_nonzero_sq2(terms):
        """terms is list of (degree, coeff). Returns count of nonzero in square."""
        max_deg = 2 * max(d for d, c in terms)
        sq = [0] * (max_deg + 1)
        n = len(terms)
        for i in range(n):
            di, ci = terms[i]
            for j in range(i, n):
                dj, cj = terms[j]
                deg = di + dj
                if i == j:
                    sq[deg] += ci * cj
                else:
                    sq[deg] += 2 * ci * cj
        return sum(1 for c in sq if c != 0)
    
    def local_search2(terms, max_iters=3000):
        current_score = count_nonzero_sq2(terms)
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            terms = list(terms)  # copy
            move_type = random.random()
            if move_type < 0.5:
                # Change coefficient
                idx = random.randint(0, 11)
                old_c = terms[idx][1]
                new_c = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -8, 8, -12, 12, -16, 16, -24, 24, -32, 32, -48, 48, -64, 64])
                if new_c == old_c or new_c == 0:
                    continue
                terms[idx] = (terms[idx][0], new_c)
                new_score = count_nonzero_sq2(terms)
                if new_score < current_score:
                    current_score = new_score
                elif new_score == current_score:
                    pass  # accept
                else:
                    terms[idx] = (terms[idx][0], old_c)
            elif move_type < 0.8:
                # Move degree
                idx = random.randint(0, 11)
                old_d = terms[idx][0]
                new_d = random.randint(0, 40)
                if new_d == old_d:
                    continue
                if any(t[0] == new_d for i, t in enumerate(terms) if i != idx):
                    continue
                terms[idx] = (new_d, terms[idx][1])
                terms.sort(key=lambda x: x[0])
                new_score = count_nonzero_sq2(terms)
                if new_score < current_score:
                    current_score = new_score
                elif new_score == current_score:
                    pass
                else:
                    terms[idx] = (old_d, terms[idx][1])
                    terms.sort(key=lambda x: x[0])
            else:
                # Add/remove/restructure: try a small random perturbation
                # Change 2 coefficients
                i, j = random.sample(range(12), 2)
                old_ci, old_cj = terms[i][1], terms[j][1]
                new_ci = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -6, 6, -8, 8, -12, 12, -16, 16, -32, 32, -64, 64])
                new_cj = random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -6, 6, -8, 8, -12, 12, -16, 16, -32, 32, -64, 64])
                if new_ci == 0 or new_cj == 0:
                    continue
                terms[i] = (terms[i][0], new_ci)
                terms[j] = (terms[j][0], new_cj)
                new_score = count_nonzero_sq2(terms)
                if new_score < current_score:
                    current_score = new_score
                elif new_score == current_score:
                    pass
                else:
                    terms[i] = (terms[i][0], old_ci)
                    terms[j] = (terms[j][0], old_cj)
        
        return terms, current_score
    
    # Restart-based search
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if r == 0:
            # Use best structured initial
            if best_result is not None:
                # Extract terms from best_result
                terms = [(i, best_result[i]) for i in range(len(best_result)) if best_result[i] != 0]
                if len(terms) == 12:
                    terms, score = local_search2(terms, max_iters=5000)
                    if score < best_score:
                        best_score = score
                        best_result = make_poly([t[0] for t in terms], [t[1] for t in terms])
                        continue
        
        # Random initial
        degrees = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -8, 8, -12, 12, -16, 16, -24, 24, -32, 32, -48, 48, -64, 64]) for _ in range(12)]
        terms = list(zip(degrees, coeffs))
        terms.sort(key=lambda x: x[0])
        
        terms, score = local_search2(terms, max_iters=2000)
        if score < best_score:
            best_score = score
            best_result = make_poly([t[0] for t in terms], [t[1] for t in terms])
    
    # Verify and return
    if best_result is None:
        # Fallback
        best_result = [0] * 41
        for i in range(12):
            best_result[i] = 1
        return best_result
    
    # Validate
    assert len(best_result) <= 41
    assert best_result[-1] != 0
    assert all(isinstance(c, int) for c in best_result)
    assert all(-64 <= c <= 64 for c in best_result)
    assert sum(1 for c in best_result if c != 0) == 12
    
    return best_result
# EVOLVE-BLOCK-END
