import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def count_nonzero(poly):
        return np.count_nonzero(poly)

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def validate(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if count_nonzero(coeffs) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    def try_update(coeffs, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            best_poly = coeffs.copy()

    # Phase 1: Constraint-Satisfaction Backtracking Search
    # We model the 12-term polynomial P(x) = sum_{k=0}^{11} a_k * x^{p_k}
    # where p_0 < p_1 < ... < p_11 are distinct integers in [0, 40].
    # The square Q(x) = P(x)^2 has coefficients q_d = sum_{i+j=d} a_i a_j.
    # We want to minimize the number of nonzero q_d.
    #
    # Key structural change: Instead of heuristic local search on a fixed support,
    # we backtracking-search over the support positions AND coefficients simultaneously,
    # using the linear constraints from the convolution to prune early.
    
    def backtrack_search(max_depth, time_limit):
        """
        Backtracking search over support positions and coefficients.
        We build the polynomial term by term.
        """
        nonlocal best_score, best_poly
        
        # We'll search for polynomials with support in a specific range to keep it tractable.
        # Let's focus on supports within [0, 20] first, then expand if time permits.
        
        def search(current_positions, current_coeffs, idx, remaining_terms):
            nonlocal best_score, best_poly
            
            if time.time() > time_limit:
                return False
            
            if remaining_terms == 0:
                # Evaluate the full square
                if len(current_positions) == 12:
                    coeffs = np.zeros(max(current_positions) + 1, dtype=np.int64)
                    for pos, val in zip(current_positions, current_coeffs):
                        coeffs[pos] = val
                    if coeffs[-1] != 0:
                        score = evaluate(coeffs)
                        try_update(coeffs, score)
                        if best_score <= 14:
                            return True
                return False
            
            # Heuristic: try to place the next term
            # The next position must be greater than the last position
            last_pos = current_positions[-1] if current_positions else -1
            
            # Determine the range of possible next positions
            # We need 12 terms total, so we need to leave room for the remaining ones.
            # Max position is 40.
            min_next = last_pos + 1
            max_next = 40 - (remaining_terms - 1)  # Leave room for rest
            
            if min_next > max_next:
                return False
            
            # To keep search space manageable, we restrict the gap between terms
            # or the total span. Let's try small gaps first.
            # Also, we can fix the first term to be at position 0 with coefficient 1
            # by symmetry (shifting and scaling don't change the number of nonzero terms
            # in the square, only their positions and values, but the count is invariant
            # under monomial multiplication and scaling by nonzero constants).
            
            # Actually, scaling by a constant c scales all square coefficients by c^2,
            # so zero/nonzero status is unchanged. Shifting by x^k shifts the square by x^{2k},
            # so zero/nonzero status is unchanged.
            # Thus, we can WLOG set the first term to be at position 0 with coefficient 1.
            
            if idx == 0:
                # First term: position 0, coefficient 1
                new_positions = current_positions + [0]
                new_coeffs = current_coeffs + [1]
                return search(new_positions, new_coeffs, idx + 1, remaining_terms - 1)
            
            # For subsequent terms, we try positions and coefficients.
            # To prune, we can check if the partial square already has too many nonzeros
            # to beat the current best. But this is expensive.
            
            # Instead, let's use a smarter heuristic:
            # Try positions that create collisions with existing positions.
            # A collision at degree d means there are multiple pairs (i,j) with p_i + p_j = d.
            
            # Let's generate candidate next positions
            candidate_positions = []
            for p in range(min_next, max_next + 1):
                candidate_positions.append(p)
            
            # Sort candidates by "collision potential" with existing positions
            # Count how many existing positions q satisfy that p+q is also a sum of two existing positions
            # or will be a sum involving p.
            
            def collision_score(p):
                score = 0
                for q in current_positions:
                    s = p + q
                    # Check if s can be formed by two other existing positions
                    for i in range(len(current_positions)):
                        for j in range(i+1, len(current_positions)):
                            if current_positions[i] + current_positions[j] == s:
                                score += 1
                return score
            
            candidate_positions.sort(key=lambda p: collision_score(p), reverse=True)
            
            # Limit the number of candidates to keep search tractable
            # Try at most 5 best candidates
            for p in candidate_positions[:5]:
                # Try a few coefficient values
                # Small coefficients are more likely to allow cancellations
                for c in [1, -1, 2, -2, 3, -3]:
                    new_positions = current_positions + [p]
                    new_coeffs = current_coeffs + [c]
                    
                    # Quick pruning: check if the partial square has too many nonzeros
                    # We can compute the square of the partial polynomial
                    # and see how many nonzeros it has.
                    # If it already exceeds best_score, we can prune.
                    
                    partial_coeffs = np.zeros(max(new_positions) + 1, dtype=np.int64)
                    for pos, val in zip(new_positions, new_coeffs):
                        partial_coeffs[pos] = val
                    
                    partial_sq = np.convolve(partial_coeffs, partial_coeffs)
                    partial_nonzero = np.count_nonzero(partial_sq)
                    
                    # The final square will have at least partial_nonzero nonzeros
                    # (actually, it could have more, but not fewer, since adding terms
                    # can only add to the sums, though cancellations can occur)
                    # Wait, adding terms can cause cancellations, so the final count
                    # could be less than partial_nonzero. So this pruning is not valid.
                    
                    # Instead, let's just continue the search
                    if search(new_positions, new_coeffs, idx + 1, remaining_terms - 1):
                        return True
                    
                    if time.time() > time_limit:
                        return True
            
            return False
        
        return search([], [], 0, 12)
    
    # Run backtracking search with a time limit
    if time.time() < deadline - 20:
        backtrack_search(12, deadline - 20)
    
    # Phase 2: Local Search Refinement
    # If backtracking found a good solution, refine it with local search.
    if best_poly is not None and time.time() < deadline - 5:
        curr_c = best_poly.copy()
        curr_score = best_score
        
        no_improvement = 0
        max_no_improvement = 10000
        temperature = 1.0
        
        while time.time() < deadline - 5 and no_improvement < max_no_improvement:
            # Pick a random index to modify
            if len(curr_c) > 0:
                idx = random.randint(0, len(curr_c) - 1)
                new_val = random.randint(-8, 8)
                if new_val == 0:
                    new_val = 1 if random.random() < 0.5 else -1
                
                if curr_c[idx] == new_val:
                    continue
                
                old_val = curr_c[idx]
                curr_c[idx] = new_val
                
                # Check constraints
                valid = True
                if len(curr_c) > 41:
                    valid = False
                if len(curr_c) == 0:
                    valid = False
                if curr_c[-1] == 0:
                    valid = False
                if not np.all((curr_c >= -64) & (curr_c <= 64)):
                    valid = False
                if np.count_nonzero(curr_c) != 12:
                    valid = False
                
                if valid:
                    new_score = evaluate(curr_c)
                    if new_score < curr_score:
                        curr_score = new_score
                        no_improvement = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr_c.copy()
                            if best_score <= 14:
                                break
                    else:
                        delta = new_score - curr_score
                        if random.random() < np.exp(-delta / max(temperature, 0.1)):
                            curr_score = new_score
                            no_improvement = 0
                            if curr_score < best_score:
                                best_score = curr_score
                                best_poly = curr_c.copy()
                        else:
                            curr_c[idx] = old_val
                            no_improvement += 1
                else:
                    curr_c[idx] = old_val
                    no_improvement += 1
            
            temperature *= 0.999
    
    # Phase 3: Try some additional structured candidates
    if time.time() < deadline - 5:
        additional_candidates = [
            [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
            [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
            [1, -2, 1, 2, -1, -2, 1, -2, 1, 2, -1, 2],
            [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
            [2, 1, -1, -2, 1, -1, 2, -1, 1, -2, 1, -1],
            [1, 2, 1, -1, -2, -1, 1, 2, 1, -1, -2, -1],
        ]
        
        for c in additional_candidates:
            if time.time() > deadline - 5:
                break
            c_arr = np.array(c, dtype=np.int64)
            score = evaluate(c_arr)
            if score < best_score:
                best_score = score
                best_poly = c_arr.copy()
    
    # Ensure final coefficient is nonzero
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)
    
    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]
    
    return best_poly.tolist()
