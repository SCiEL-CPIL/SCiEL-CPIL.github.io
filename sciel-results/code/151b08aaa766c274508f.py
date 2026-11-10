"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: 55s out of 60s/90s limit
    
    def count_nonzero(poly):
        return int(np.count_nonzero(poly))
    
    def poly_square_dense(dense):
        """Square a dense polynomial array."""
        return np.convolve(dense, dense, mode='full')
    
    def terms_to_dense(terms):
        """Convert list of (pos, val) to dense array."""
        if not terms:
            return np.array([0], dtype=np.int64)
        max_deg = max(pos for pos, _ in terms)
        dense = np.zeros(max_deg + 1, dtype=np.int64)
        for pos, val in terms:
            dense[pos] = val
        return dense
    
    def evaluate_terms(terms):
        """Evaluate number of nonzero terms in square of given terms."""
        if len(terms) != 12:
            return 1000
        positions = [pos for pos, _ in terms]
        if len(set(positions)) != 12:
            return 1000
        for pos, val in terms:
            if val == 0 or abs(val) > 64 or pos > 40:
                return 1000
        dense = terms_to_dense(terms)
        sq = poly_square_dense(dense)
        return count_nonzero(sq)
    
    def terms_to_coeff_list(terms):
        """Convert terms to coefficient list in ascending degree."""
        if not terms:
            return [1]
        max_pos = max(pos for pos, _ in terms)
        coeff_list = [0] * (max_pos + 1)
        for pos, val in terms:
            coeff_list[pos] = val
        if len(coeff_list) > 41:
            coeff_list = coeff_list[:41]
        while coeff_list and coeff_list[-1] == 0:
            coeff_list.pop()
        if not coeff_list:
            coeff_list = [1]
        return coeff_list
    
    # Strategy: Try structured polynomials that exploit cancellation
    # Key insight: If P(x) = sum a_i x^{p_i}, then P(x)^2 = sum a_i^2 x^{2*p_i} + 2*sum_{i<j} a_i*a_j x^{p_i+p_j}
    # To minimize nonzero terms, we want many cross-terms to cancel or coincide.
    
    best_score = 23  # Upper bound: max possible is 2*12-1=23
    best_terms = None
    
    # Phase 1: Try polynomials with specific structural patterns
    # Pattern 1: Arithmetic progression positions
    for n in range(12):
        for start in range(0, 41 - n * 11):
            if time.time() > deadline:
                break
            positions = [start + i * 11 for i in range(12)]
            if positions[-1] > 40:
                continue
            # Try all sign patterns (2^12 = 4096, too many)
            # Instead, try a few representative sign patterns
            for sign_pattern in [
                [1]*12,
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
            ]:
                terms = [(positions[i], sign_pattern[i]) for i in range(12)]
                score = evaluate_terms(terms)
                if score < best_score:
                    best_score = score
                    best_terms = terms[:]
                if best_score <= 12:
                    break
            if best_score <= 12:
                break
        if best_score <= 12:
            break
    
    # Phase 2: Try palindromic structures
    if time.time() < deadline:
        for center in range(0, 41):
            if time.time() > deadline:
                break
            # Try symmetric positions around center
            offsets = list(range(0, 6))  # 6 positive offsets + center = 7, need 12
            # Better: use 12 positions symmetrically
            for max_off in range(1, 21):
                positions = []
                for i in range(6):
                    if center - i >= 0 and center + i <= 40:
                        positions.append(center - i)
                        positions.append(center + i)
                if len(positions) == 12:
                    positions.sort()
                    for signs in [
                        [1]*12,
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                    ]:
                        terms = [(positions[i], signs[i]) for i in range(12)]
                        score = evaluate_terms(terms)
                        if score < best_score:
                            best_score = score
                            best_terms = terms[:]
                        if best_score <= 12:
                            break
                if best_score <= 12:
                    break
            if best_score <= 12:
                break
    
    # Phase 3: Local search from best found so far, with smarter moves
    if best_terms is None:
        # Fallback: random initial
        positions = random.sample(range(41), 12)
        values = [random.choice([1, -1]) for _ in range(12)]
        best_terms = list(zip(positions, values))
        best_score = evaluate_terms(best_terms)
    
    # Local search with position and value optimization
    current_terms = best_terms[:]
    current_score = best_score
    
    def neighbor_position(terms):
        """Move one term to a new position."""
        terms = list(terms)
        idx = random.randint(0, 11)
        old_pos = terms[idx][0]
        new_pos = random.randint(0, 40)
        if new_pos == old_pos:
            return terms
        for i, (pos, val) in enumerate(terms):
            if i != idx and pos == new_pos:
                return terms
        terms[idx] = (new_pos, terms[idx][1])
        return terms
    
    def neighbor_value(terms):
        """Change value of one term."""
        terms = list(terms)
        idx = random.randint(0, 11)
        new_val = random.choice([1, -1]) * random.randint(1, 64)
        terms[idx] = (terms[idx][0], new_val)
        return terms
    
    def neighbor_swap(terms):
        """Swap positions of two terms."""
        terms = list(terms)
        i, j = random.sample(range(12), 2)
        terms[i], terms[j] = (terms[j][0], terms[i][1]), (terms[i][0], terms[j][1])
        return terms
    
    local_iter = 0
    max_local_iter = 50000
    while local_iter < max_local_iter and time.time() < deadline:
        local_iter += 1
        move_type = random.random()
        if move_type < 0.4:
            new_terms = neighbor_position(current_terms)
        elif move_type < 0.7:
            new_terms = neighbor_value(current_terms)
        else:
            new_terms = neighbor_swap(current_terms)
        
        new_score = evaluate_terms(new_terms)
        if new_score < current_score:
            current_terms = new_terms
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_terms = current_terms[:]
                if best_score <= 12:
                    break
        elif new_score == current_score and random.random() < 0.3:
            # Accept equal to escape local minima
            current_terms = new_terms
    
    # Phase 4: Try to further optimize by adjusting values to create more cancellations
    if time.time() < deadline and best_score > 12:
        # For the best polynomial, try to adjust values to create more zero terms in square
        current_terms = best_terms[:]
        current_score = best_score
        
        for _ in range(10000):
            if time.time() > deadline:
                break
            # Pick a term and try different values
            idx = random.randint(0, 11)
            old_val = current_terms[idx][1]
            # Try small values first
            candidates = [v for v in range(-64, 65) if v != 0 and v != old_val]
            random.shuffle(candidates)
            improved = False
            for new_val in candidates[:20]:
                test_terms = current_terms[:]
                test_terms[idx] = (current_terms[idx][0], new_val)
                test_score = evaluate_terms(test_terms)
                if test_score < current_score:
                    current_terms = test_terms
                    current_score = test_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_terms = current_terms[:]
                    break
            if best_score <= 12:
                break
            if not improved and random.random() < 0.1:
                # Restart local search from a random point
                positions = random.sample(range(41), 12)
                values = [random.choice([1, -1]) * random.randint(1, 10) for _ in range(12)]
                current_terms = list(zip(positions, values))
                current_score = evaluate_terms(current_terms)
    
    # Convert to coefficient list
    return terms_to_coeff_list(best_terms)
# EVOLVE-BLOCK-END
