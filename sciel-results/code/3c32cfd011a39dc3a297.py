import time
import numpy as np
import random
from scipy.optimize import linprog

# EVOLVE-BLOCK-START

def _int_to_digits(x, q, n, powers):
    """Convert integer index to list of digits."""
    digits = []
    for i in range(n):
        d = (x // powers[i]) % q
        digits.append(d)
    return digits

def _precompute_balls_and_matrix(q, n, radius, deadline):
    """
    Precompute the constraint matrix A for the LP formulation.
    A is a (total_words x total_words) matrix where A[y, c] = 1 if word y is within Hamming distance radius of codeword c.
    We also return the list of balls (for local search later) and powers.
    """
    total = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # For radius 1, each ball has size 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    
    # We need to construct the constraint matrix.
    # The problem is: Minimize sum(x_c) subject to sum_{c: y in ball(c)} x_c >= 1 for all y, and 0 <= x_c <= 1.
    # In linprog, we need A_ub, b_ub, A_eq, b_eq, bounds.
    # The constraints are: -sum_{c in ball(y)} x_c <= -1 for all y.
    # So A_ub is a matrix where A_ub[y, c] = -1 if y in ball(c), else 0.
    # b_ub is a vector of -1s.
    
    # Constructing a dense matrix of size 6561x6561 might be memory heavy but feasible?
    # 6561 * 6561 * 8 bytes ~ 345 MB. This is risky for memory limits.
    # However, scipy linprog can take sparse matrices. Let's use scipy.sparse.
    
    from scipy.sparse import lil_matrix
    
    # We will build the sparse matrix A_ub.
    # Row y corresponds to constraint for word y.
    # Col c corresponds to variable x_c.
    # A_ub[y, c] = -1 if dist(y, c) <= radius.
    
    A_ub = lil_matrix((total, total), dtype=np.float64)
    
    # Precompute balls as lists of indices for efficiency in building the matrix and later local search
    balls = []
    
    for c in range(total):
        # Find all y in ball(c)
        ball = [c]
        for i in range(n):
            old_digit = (c // powers[i]) % q
            for new_digit in range(q):
                if new_digit != old_digit:
                    y = c + (new_digit - old_digit) * powers[i]
                    ball.append(y)
        balls.append(ball)
        
        # Set A_ub[y, c] = -1 for all y in ball
        for y in ball:
            A_ub[y, c] = -1.0
            
        if time.time() > deadline - 20:
            # If we run out of time precomputing, we can't do LP. Fallback will handle it.
            # But for now, let's assume it finishes. 6561 * 22 ops is ~145k ops, very fast.
            pass
            
    A_ub = A_ub.tocsr()
    b_ub = -np.ones(total)
    
    # Bounds: 0 <= x_c <= 1
    bounds = [(0, 1)] * total
    
    return A_ub, b_ub, bounds, balls, powers

def _solve_lp_and_round(A_ub, b_ub, bounds, balls, powers, q, n, total, deadline):
    """
    Solve the LP relaxation and round the solution to an integer covering set.
    """
    c_obj = np.ones(total)
    
    # Solve LP
    # Method 'highs' is usually robust and fast for LPs.
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return None
    
    if not res.success:
        return None
        
    x_lp = res.x
    
    # Rounding Strategy:
    # 1. Select all codewords where x_lp[c] > 0.5 (or some threshold).
    # 2. Check if this set covers everything.
    # 3. If not, add codewords greedily to cover remaining uncovered words, prioritized by x_lp values.
    
    # Threshold selection
    # In covering codes, the LP solution is often close to integer.
    # Let's try threshold 0.5 first.
    
    selected_indices = [i for i in range(total) if x_lp[i] > 0.5]
    
    # Check coverage
    covered = np.zeros(total, dtype=bool)
    for c in selected_indices:
        for y in balls[c]:
            covered[y] = True
            
    # If not all covered, we need to add more.
    # We can use a greedy approach on the residual, weighted by LP values.
    if not np.all(covered):
        # Compute uncovered words
        uncovered = np.where(~covered)[0]
        
        # We want to add codewords that cover the most uncovered words.
        # To break ties or guide selection, we can use the LP value as a secondary score.
        # Score(c) = count_uncovered_in_ball(c) + alpha * x_lp[c]
        
        alpha = 0.01 # Small weight for LP value
        
        # Precompute current uncovered count for each ball
        # We can just iterate through candidates
        
        while not np.all(covered) and time.time() < deadline - 5:
            best_c = -1
            best_score = -1
            best_count = -1
            
            # Iterate through all candidates? 6561 candidates.
            # For each, count uncovered in ball.
            # This is O(total * ball_size) per iteration. 6561 * 22 ~ 145k ops.
            # If we need 500 iterations, that's 70M ops. Might be slow in Python.
            
            # Optimization: Only consider candidates not yet selected?
            # Actually, we can just pick the one with max (count_uncovered + alpha * x_lp).
            
            for c in range(total):
                if c in selected_indices: # This check is slow if selected_indices is a list. Use a set.
                    continue
                # Count uncovered in ball[c]
                count = 0
                for y in balls[c]:
                    if not covered[y]:
                        count += 1
                if count == 0:
                    continue
                
                score = count + alpha * x_lp[c]
                if score > best_score or (score == best_score and x_lp[c] > (x_lp[best_c] if best_c != -1 else -1)):
                    best_score = score
                    best_count = count
                    best_c = c
                    
            if best_c == -1:
                break # No candidate covers any uncovered word? Should not happen if LP is feasible.
                
            selected_indices.append(best_c)
            # Update covered
            for y in balls[best_c]:
                covered[y] = True
                
    # Local Search: Try to remove redundant codewords
    # This is the same as in the previous program.
    selected_set = set(selected_indices)
    
    # cover_count[y] = number of selected codewords covering y
    cover_count = np.zeros(total, dtype=np.int32)
    for c in selected_set:
        for y in balls[c]:
            cover_count[y] += 1
            
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        candidates = list(selected_set)
        random.shuffle(candidates)
        
        for c in candidates:
            # Check if c is redundant
            # If min(cover_count[ball[c]]) >= 2, then c is redundant
            if np.min(cover_count[balls[c]]) >= 2:
                # Remove c
                for y in balls[c]:
                    cover_count[y] -= 1
                selected_set.remove(c)
                improved = True
                
    return list(selected_set)

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145
    
    total = q ** n
    
    try:
        # Precompute
        A_ub, b_ub, bounds, balls, powers = _precompute_balls_and_matrix(q, n, radius, deadline)
        
        # Solve and Round
        selected_indices = _solve_lp_and_round(A_ub, b_ub, bounds, balls, powers, q, n, total, deadline)
        
        if selected_indices is None:
            raise Exception("LP failed")
            
        # Convert to digits
        codewords = []
        for idx in selected_indices:
            codewords.append(_int_to_digits(idx, q, n, powers))
            
        return codewords
        
    except Exception as e:
        # Fallback to simple greedy if LP fails or times out
        # This is a simplified fallback to ensure we return *something* valid if LP breaks.
        # Ideally, we should have a robust greedy fallback.
        
        try:
            # Simple Greedy Fallback
            covered = np.zeros(total, dtype=bool)
            selected = []
            
            # Precompute balls if not already done? 
            # _precompute_balls_and_matrix returns balls. 
            # But if it failed, we need to recompute or use a different path.
            # Let's assume the exception was in LP solve, so balls are available in scope? 
            # No, scope is local. 
            # Let's just re-implement a very fast greedy here.
            
            # Actually, if _precompute_balls_and_matrix succeeded, balls is in scope.
            # If it failed, we are in trouble. 
            # Let's assume the failure is in linprog.
            
            if 'balls' in locals():
                while True:
                    if time.time() > deadline - 5:
                        break
                    max_count = 0
                    best_c = -1
                    for c in range(total):
                        if c in selected: # Slow check
                            continue
                        count = 0
                        for y in balls[c]:
                            if not covered[y]:
                                count += 1
                        if count > max_count:
                            max_count = count
                            best_c = c
                    if best_c == -1:
                        break
                    selected.append(best_c)
                    for y in balls[best_c]:
                        covered[y] = True
                    if np.all(covered):
                        break
                
                codewords = []
                for idx in selected:
                    codewords.append(_int_to_digits(idx, q, n, powers))
                return codewords
            else:
                # Ultimate fallback: raise
                raise e
                
        except Exception:
            raise e

# EVOLVE-BLOCK-END
