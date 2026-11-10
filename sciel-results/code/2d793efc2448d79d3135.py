"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: 55s out of 60s/90s limit
    
    def count_nonzero(poly):
        return np.count_nonzero(poly)
    
    def poly_square(coeffs):
        # coeffs is a list of (position, value) tuples
        # Convert to dense array for convolution
        if not coeffs:
            return np.array([0])
        max_deg = max(pos for pos, _ in coeffs)
        dense = np.zeros(max_deg + 1, dtype=np.int64)
        for pos, val in coeffs:
            dense[pos] = val
        # Square via convolution
        result = np.convolve(dense, dense, mode='full')
        return result
    
    def evaluate(terms):
        """terms: list of (position, value) tuples, exactly 12 nonzero"""
        # Validation is assumed to be handled by construction in the search loop
        # for speed, but we ensure validity here.
        if len(terms) != 12:
            return 1000
        positions = [pos for pos, _ in terms]
        if len(set(positions)) != 12:
            return 1000
        for pos, val in terms:
            if val == 0 or abs(val) > 64 or pos > 40:
                return 1000
        
        sq = poly_square(terms)
        return count_nonzero(sq)
    
    def random_initial():
        """Generate a random valid 12-term polynomial"""
        positions = random.sample(range(41), 12)
        values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
        return list(zip(positions, values))
    
    def neighbor(terms):
        """Generate a neighboring solution by modifying one term"""
        terms = list(terms)
        move_type = random.random()
        
        if move_type < 0.5:
            # Change position of one term
            idx = random.randint(0, 11)
            old_pos = terms[idx][0]
            new_pos = random.randint(0, 40)
            if new_pos == old_pos:
                return terms
            # Check no collision
            for i, (pos, val) in enumerate(terms):
                if i != idx and pos == new_pos:
                    return terms
            terms[idx] = (new_pos, terms[idx][1])
        elif move_type < 0.8:
            # Change value of one term
            idx = random.randint(0, 11)
            new_val = random.choice([1, -1]) * random.randint(1, 64)
            terms[idx] = (terms[idx][0], new_val)
        else:
            # Change sign of one term
            idx = random.randint(0, 11)
            terms[idx] = (terms[idx][0], -terms[idx][1])
        
        return terms

    best_terms = random_initial()
    best_score = evaluate(best_terms)
    
    # Random Restart with Local Search
    # We use NumPy for speed, so we can afford more iterations.
    max_iterations = 1000000  # High cap, but time is the real limit
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Random restart or continue from current best?
        # Let's mix: 50% restart, 50% local search from best
        if random.random() < 0.5:
            current_terms = random_initial()
            current_score = evaluate(current_terms)
        else:
            current_terms = best_terms[:]
            current_score = best_score
            
        # Local search phase: try to improve current_terms
        improved = True
        local_iter = 0
        while improved and local_iter < 20 and time.time() < deadline:
            improved = False
            local_iter += 1
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_terms = neighbor(current_terms)
                new_score = evaluate(new_terms)
                if new_score < current_score:
                    current_terms = new_terms
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_terms = current_terms[:]
                        if best_score <= 12:
                            break
            if best_score <= 12:
                break
        
        if best_score <= 12:
            break

    # Convert best terms back to coefficient list
    if not best_terms:
        return [1] # Fallback, though should not happen
        
    max_pos = max(pos for pos, _ in best_terms)
    coeff_list = [0] * (max_pos + 1)
    for pos, val in best_terms:
        coeff_list[pos] = val
    
    # Ensure length <= 41 and nonzero final coefficient
    if len(coeff_list) > 41:
        coeff_list = coeff_list[:41]
    if coeff_list[-1] == 0:
        # Find last nonzero
        while coeff_list and coeff_list[-1] == 0:
            coeff_list.pop()
        if not coeff_list:
            coeff_list = [1]
            
    return coeff_list
# EVOLVE-BLOCK-END
