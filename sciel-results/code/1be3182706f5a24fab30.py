import numpy as np
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def evaluate(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))
    
    def validate(coeffs):
        if len(coeffs) != 12:
            return False
        if coeffs[-1] == 0 or coeffs[0] == 0:
            return False
        if not all(isinstance(c, int) for c in coeffs):
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True
    
    best_score = float('inf')
    best_poly = None
    
    random.seed(42)
    
    # Precompute pair contributions for each square degree
    pair_contribs = {k: [] for k in range(23)}
    for i in range(12):
        for j in range(12):
            pair_contribs[i + j].append((i, j))
    
    def compute_square(coeffs):
        sq = [0] * 23
        for i in range(12):
            ci = coeffs[i]
            for j in range(12):
                sq[i + j] += ci * coeffs[j]
        return sq
    
    def count_nz(sq):
        return sum(1 for x in sq if x != 0)
    
    def try_zero_term(coeffs, k):
        """Try to zero out sq[k] by changing one coefficient.
        Returns (index, new_val, new_score) or None."""
        sq = compute_square(coeffs)
        if sq[k] == 0:
            return None
        
        current_score = count_nz(sq)
        
        for i in range(12):
            # Determine A, B, C such that sq[k] = A*c[i]^2 + B*c[i] + C
            A = 0
            B = 0
            C = sq[k]
            
            for (a, b) in pair_contribs[k]:
                if a == i and b == i:
                    A = 1
                    C -= coeffs[i] * coeffs[i]
                elif a == i or b == i:
                    other = b if a == i else a
                    B += coeffs[other]
                    C -= coeffs[i] * coeffs[other]
            
            # Solve A*x^2 + B*x + C = 0 for integer x in [-64,64], x != 0
            solutions = set()
            if A == 0:
                if B != 0 and C % B == 0:
                    x = -C // B
                    if -64 <= x <= 64 and x != 0:
                        solutions.add(x)
            else:
                disc = B * B - 4 * A * C
                if disc >= 0:
                    sqrt_disc = int(round(np.sqrt(disc)))
                    for s in [sqrt_disc, -sqrt_disc]:
                        if s * s == disc:
                            for sign in [1, -1]:
                                num = -B + sign * s
                                den = 2 * A
                                if den != 0 and num % den == 0:
                                    x = num // den
                                    if -64 <= x <= 64 and x != 0:
                                        solutions.add(x)
            
            for new_val in solutions:
                if new_val == coeffs[i]:
                    continue
                old_val = coeffs[i]
                coeffs[i] = new_val
                new_sq = compute_square(coeffs)
                new_score = count_nz(new_sq)
                coeffs[i] = old_val
                
                if new_score < current_score:
                    return (i, new_val, new_score)
        
        return None
    
    def local_search(curr, curr_score, max_iters=5000):
        """Iteratively try to zero out nonzero square terms."""
        no_improve = 0
        max_no_improve = 200
        
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            sq = compute_square(curr)
            nonzero_k = [k for k in range(23) if sq[k] != 0]
            
            if not nonzero_k:
                break
            
            # Pick a random nonzero term to target
            random.shuffle(nonzero_k)
            improved = False
            
            for k in nonzero_k:
                result = try_zero_term(curr, k)
                if result is not None:
                    idx, new_val, new_score = result
                    curr[idx] = new_val
                    curr_score = new_score
                    improved = True
                    no_improve = 0
                    break
            
            if not improved:
                # Try random single-coordinate perturbation
                idx = random.randint(0, 11)
                old_val = curr[idx]
                
                if random.random() < 0.6:
                    new_val = random.choice([-2, -1, 1, 2, 3, -3])
                else:
                    new_val = random.randint(-64, 64)
                
                if new_val == 0:
                    new_val = 1
                if new_val == old_val:
                    continue
                
                curr[idx] = new_val
                new_score = evaluate(curr)
                
                if new_score < curr_score:
                    curr_score = new_score
                    no_improve = 0
                else:
                    curr[idx] = old_val
                    no_improve += 1
            
            if no_improve > max_no_improve:
                break
        
        return curr, curr_score
    
    # Phase 1: Multiple random restarts with targeted local search
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize with structured random pattern
        curr = [random.choice([-1, 1]) for _ in range(12)]
        curr_score = evaluate(curr)
        
        # Apply local search
        curr, curr_score = local_search(curr, curr_score)
        
        if curr_score < best_score:
            best_score = curr_score
            best_poly = curr.copy()
    
    # Phase 2: Focused refinement on best solution
    if best_poly is not None and time.time() < deadline - 5:
        curr = best_poly.copy()
        curr_score = best_score
        
        # More aggressive local search with larger neighborhood
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            sq = compute_square(curr)
            nonzero_k = [k for k in range(23) if sq[k] != 0]
            
            if not nonzero_k:
                break
            
            improved = False
            random.shuffle(nonzero_k)
            
            for k in nonzero_k:
                result = try_zero_term(curr, k)
                if result is not None:
                    idx, new_val, new_score = result
                    curr[idx] = new_val
                    curr_score = new_score
                    improved = True
                    no_improve = 0
                    break
            
            if not improved:
                # Try multi-coordinate changes: change two coefficients
                idx1 = random.randint(0, 11)
                idx2 = random.randint(0, 11)
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % 12
                
                old1, old2 = curr[idx1], curr[idx2]
                
                v1 = random.choice([-2, -1, 1, 2, 3, -3, 5, -5])
                v2 = random.choice([-2, -1, 1, 2, 3, -3, 5, -5])
                if v1 == 0:
                    v1 = 1
                if v2 == 0:
                    v2 = 1
                
                curr[idx1] = v1
                curr[idx2] = v2
                new_score = evaluate(curr)
                
                if new_score < curr_score:
                    curr_score = new_score
                    no_improve = 0
                else:
                    curr[idx1] = old1
                    curr[idx2] = old2
                    no_improve += 1
            
            if curr_score < best_score:
                best_score = curr_score
                best_poly = curr.copy()
    
    # Phase 3: Final hill climbing on best
    if best_poly is not None and time.time() < deadline - 2:
        curr = best_poly.copy()
        curr_score = best_score
        
        while time.time() < deadline - 1:
            improved = False
            for idx in range(12):
                old_val = curr[idx]
                for delta in [-3, -2, -1, 1, 2, 3, 5, -5, 8, -8]:
                    new_val = old_val + delta
                    if new_val == 0 or not (-64 <= new_val <= 64):
                        continue
                    curr[idx] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                        break
                curr[idx] = old_val
            
            if not improved:
                break
    
    if best_poly is None:
        best_poly = [1] * 12
    
    if not validate(best_poly):
        best_poly = [1] * 12
    
    return best_poly
