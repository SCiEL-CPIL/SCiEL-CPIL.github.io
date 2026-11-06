import time
import random
import itertools
import numpy as np
from scipy.optimize import linprog

def _count_nonzero(coeffs):
    """Count nonzero terms in a coefficient list (ascending degree)."""
    if not coeffs:
        return 0
    # Trim trailing zeros for accurate degree, but count all nonzeros
    return sum(1 for c in coeffs if c != 0)

def _square_count(coeffs):
    """Compute the number of nonzero terms in the exact square of the polynomial."""
    # Use numpy for fast convolution
    sq = np.convolve(coeffs, coeffs)
    return int(np.count_nonzero(sq))

def _construct_from_supports(supports, coeffs):
    """Construct polynomial from supports and coefficients, return list."""
    # Merge supports and coeffs into a single list
    # supports is a list of exponents, coeffs is a list of values
    if not supports:
        return [0]
    max_deg = max(supports)
    res = [0] * (max_deg + 1)
    for e, c in zip(supports, coeffs):
        res[e] = c
    return res

def _find_sparse_solution(deadline):
    """
    Search for a 12-term polynomial with minimal square sparsity.
    Strategy:
    1. Generate random 12-element exponent sets (supports).
    2. Analyze the sumset E+E to identify collision classes.
    3. If the number of distinct sums is small, it's a candidate for high cancellation.
    4. For a candidate support, set up an ILP/LP to find integer coefficients in [-64, 64]
       that minimize the number of nonzero terms in the square.
       Note: Minimizing sparsity is non-convex. We will use a heuristic:
       - Try to find coefficients such that P^2 is a monomial or binomial.
       - This is equivalent to finding a "perfect square" polynomial or similar.
       
    However, a simpler structural approach:
    Look for supports that are "sum-free" or have very few collisions.
    If E+E has size 12, then P^2 has at most 12 terms (if no cancellations).
    If E+E has size < 12, cancellations are possible.
    
    Actually, the minimum possible nonzero terms in P^2 for a 12-term P is 1 (if P is a monomial, but P has 12 terms).
    Wait, P has 12 nonzero terms.
    If P = x^12 - 1, P^2 = x^24 - 2x^12 + 1 (3 terms).
    If P = x^12 + x^6 + 1, P^2 = x^24 + 2x^18 + 3x^12 + 2x^6 + 1 (5 terms).
    
    Let's search for supports that allow P^2 to have very few terms.
    A known identity: (a x^k + b x^j + c x^i)^2 can have few terms if coefficients cancel.
    Example: (x^2 - x + 1)^2 = x^4 - 2x^3 + 3x^2 - 2x + 1 (5 terms).
    
    Let's try to find a 12-term polynomial P such that P^2 has <= 12 terms.
    The seed returns 12 (for [1]*12? No, [1]*12 is 12 terms of 1. Square is 1,2,...,12,...,1. 23 terms.
    Wait, the seed program returns [1]*12.
    P(x) = 1 + x + ... + x^11.
    P(x)^2 = 1 + 2x + ... + 11x^10 + 12x^11 + 11x^12 + ... + x^22.
    Number of nonzero terms = 23.
    Score = 12/23 ≈ 0.52? No, the score is seed/value.
    The task says: "The evaluator measures the raw value (lower is better). The search score is seed/value".
    If seed program returns 12 terms in square? No, seed program returns [1]*12.
    Let's check the seed program's square count.
    P = 1+x+...+x^11.
    P^2 has 23 nonzero terms.
    So value = 23.
    Score = seed_value / current_value?
    "search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    This implies: Score = (Seed_Square_Count) / (Current_Square_Count).
    If Seed_Square_Count = 23, and Current_Square_Count = 23, Score = 1.0.
    If Current_Square_Count = 12, Score = 23/12 = 1.91.
    If Current_Square_Count = 1, Score = 23/1 = 23.
    
    So we want to MINIMIZE the number of nonzero terms in P^2.
    The minimum possible is 1?
    If P^2 is a monomial, P must be a monomial (in char 0, domain).
    But P has 12 nonzero terms. So P^2 cannot be a monomial.
    Can P^2 be a binomial?
    P^2 = A x^k + B x^j.
    P = sqrt(A) x^{k/2} + i sqrt(B) x^{j/2}? No, coefficients must be integers.
    If P^2 = x^{2n} + 1, P = ? No integer polynomial solution.
    
    Let's look at the inspirations.
    Inspiration 1 score 1.5333. Value = 23 / 1.5333 ≈ 15.
    Inspiration 2 score 1.4375. Value = 23 / 1.4375 ≈ 16.
    
    We want to get as close to 12 as possible (or lower).
    The theoretical minimum for a 12-term polynomial square is not 1.
    However, we can try to find structures where many cross terms cancel.
    
    Strategy:
    1. Randomly generate 12 exponent sets.
    2. Calculate the collision structure of E+E.
    3. For each distinct sum s, the coefficient in P^2 is C_s = sum_{i+j=s} a_i a_j.
    4. We want to choose a_i in [-64, 64] \ {0} such that as many C_s are 0 as possible.
    5. This is a system of quadratic equations C_s = 0.
    6. We can use a local search / simulated annealing on the coefficients to minimize the count of non-zero C_s.
    """
    best_val = 23 # Seed value
    best_coeffs = [1] * 12
    
    # Time management
    start_time = time.time()
    
    # Precompute some known good structures?
    # Let's just do a robust local search on random supports.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Generate a random support of size 12
        # Use a wide range to allow for sparse sums
        max_deg = 100
        supports = sorted(random.sample(range(max_deg), 12))
        
        # Map exponent to index
        exp_to_idx = {e: i for i, e in enumerate(supports)}
        
        # Identify all pairs (i, j) with i <= j such that supports[i] + supports[j] is a valid sum.
        # Actually, we just need the list of (i, j) pairs that contribute to each sum.
        sum_to_pairs = {}
        for i in range(12):
            for j in range(i, 12):
                s = supports[i] + supports[j]
                if s not in sum_to_pairs:
                    sum_to_pairs[s] = []
                sum_to_pairs[s].append((i, j))
        
        # Number of distinct sums
        n_sums = len(sum_to_pairs)
        
        # If n_sums is large, it's hard to cancel many.
        # We want n_sums to be small.
        # Random supports usually have ~78 distinct sums (12*13/2).
        # We want to find supports with fewer distinct sums.
        # This happens if the support is an arithmetic progression or similar.
        
        # Let's bias the generation towards arithmetic progressions or clusters.
        # For this restart, let's try a structured support.
        if restart % 2 == 0:
            # Arithmetic progression
            a = random.randint(1, 5)
            d = random.randint(1, 10)
            supports = [a + i * d for i in range(12)]
        else:
            # Clustered
            base = random.randint(0, 50)
            supports = sorted(base + random.randint(0, 20) for _ in range(12))
            # Ensure uniqueness
            while len(set(supports)) < 12:
                supports.append(random.randint(0, 100))
            supports = sorted(set(supports))[:12]
            if len(supports) < 12:
                continue

        exp_to_idx = {e: i for i, e in enumerate(supports)}
        sum_to_pairs = {}
        for i in range(12):
            for j in range(i, 12):
                s = supports[i] + supports[j]
                if s not in sum_to_pairs:
                    sum_to_pairs[s] = []
                sum_to_pairs[s].append((i, j))
        
        # Initialize coefficients with random integers in [-10, 10]
        coeffs = [random.randint(1, 10) * (1 if random.random() > 0.5 else -1) for _ in range(12)]
        
        # Local search: Simulated Annealing
        current_val = _square_count(_construct_from_supports(supports, coeffs))
        best_local_val = current_val
        best_local_coeffs = coeffs[:]
        
        T = 5.0
        for step in range(500):
            if time.time() > deadline:
                break
                
            # Pick a random coefficient to change
            idx = random.randint(0, 11)
            old_val = coeffs[idx]
            new_val = old_val + random.choice([-2, -1, 1, 2])
            if new_val == 0:
                # Allow 0? No, exactly 12 nonzero terms in P.
                # So we must keep coeffs[idx] != 0.
                new_val = 1 if old_val > 0 else -1
            if abs(new_val) > 64:
                new_val = 64 if new_val > 0 else -64
                
            coeffs[idx] = new_val
            new_poly = _construct_from_supports(supports, coeffs)
            new_val_sq = _square_count(new_poly)
            
            delta = new_val_sq - current_val
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_val = new_val_sq
                if current_val < best_local_val:
                    best_local_val = current_val
                    best_local_coeffs = coeffs[:]
            else:
                coeffs[idx] = old_val
                
            T *= 0.95
            if T < 0.01:
                break
                
        if best_local_val < best_val:
            best_val = best_local_val
            best_coeffs = best_local_coeffs[:]
            # Update global best
            # We might want to continue refining this specific support
            # But for now, just record it.
            
            # If we found a very good one, we can try to refine it further
            if best_val <= 12:
                # Try to optimize coefficients for this specific support with more iterations
                T = 2.0
                for step in range(1000):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, 11)
                    old_val = coeffs[idx]
                    new_val = old_val + random.choice([-1, 1])
                    if new_val == 0:
                        new_val = 1 if old_val > 0 else -1
                    if abs(new_val) > 64:
                        continue
                    coeffs[idx] = new_val
                    new_poly = _construct_from_supports(supports, coeffs)
                    new_val_sq = _square_count(new_poly)
                    delta = new_val_sq - current_val
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        current_val = new_val_sq
                        if current_val < best_val:
                            best_val = current_val
                            best_coeffs = coeffs[:]
                    else:
                        coeffs[idx] = old_val
                    T *= 0.99
                    
                if best_val == 12:
                    break # Can't get better than 12? 
                    # Actually, can we get 11? 
                    # Min terms in square of 12-term poly.
                    # If P has 12 terms, P^2 has at least ... ?
                    # There is no strict lower bound of 12.
                    # But 12 is a good target.
                    
    # Final check
    final_poly = _construct_from_supports(supports, best_coeffs) if 'supports' in locals() else [1]*12
    # Wait, if we break early, supports might be from the last iteration.
    # We need to store the best supports too.
    
    # Let's refactor to store best supports and coeffs together.
    
    return final_poly

def construct():
    # Deadline guard: 60s limit, use 55s for safety
    deadline = time.time() + 55.0
    
    best_val = 23
    best_poly = [1] * 12
    
    # We will search for (supports, coeffs) pairs
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Generate support
        if restart % 3 == 0:
            # AP
            a = random.randint(1, 10)
            d = random.randint(1, 10)
            supports = [a + i * d for i in range(12)]
        elif restart % 3 == 1:
            # Random
            supports = sorted(random.sample(range(150), 12))
        else:
            # Clustered
            base = random.randint(0, 50)
            supports = []
            while len(supports) < 12:
                supports.append(base + random.randint(0, 30))
            supports = sorted(set(supports))
            if len(supports) < 12:
                continue
            supports = supports[:12]
            
        # Map
        exp_to_idx = {e: i for i, e in enumerate(supports)}
        sum_to_pairs = {}
        for i in range(12):
            for j in range(i, 12):
                s = supports[i] + supports[j]
                if s not in sum_to_pairs:
                    sum_to_pairs[s] = []
                sum_to_pairs[s].append((i, j))
                
        # Init coeffs
        coeffs = [random.randint(1, 10) * (1 if random.random() > 0.5 else -1) for _ in range(12)]
        
        # SA
        current_val = _square_count(_construct_from_supports(supports, coeffs))
        best_local_val = current_val
        best_local_coeffs = coeffs[:]
        
        T = 5.0
        for step in range(200):
            if time.time() > deadline:
                break
            idx = random.randint(0, 11)
            old_val = coeffs[idx]
            new_val = old_val + random.choice([-2, -1, 1, 2])
            if new_val == 0:
                new_val = 1 if old_val > 0 else -1
            if abs(new_val) > 64:
                new_val = 64 if new_val > 0 else -64
                
            coeffs[idx] = new_val
            new_poly = _construct_from_supports(supports, coeffs)
            new_val_sq = _square_count(new_poly)
            
            delta = new_val_sq - current_val
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_val = new_val_sq
                if current_val < best_local_val:
                    best_local_val = current_val
                    best_local_coeffs = coeffs[:]
            else:
                coeffs[idx] = old_val
                
            T *= 0.95
            if T < 0.01:
                break
                
        if best_local_val < best_val:
            best_val = best_local_val
            best_poly = _construct_from_supports(supports, best_local_coeffs)
            
            # If we find a very good one, refine
            if best_val <= 14:
                T = 2.0
                for step in range(500):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, 11)
                    old_val = best_local_coeffs[idx]
                    new_val = old_val + random.choice([-1, 1])
                    if new_val == 0:
                        new_val = 1 if old_val > 0 else -1
                    if abs(new_val) > 64:
                        continue
                    best_local_coeffs[idx] = new_val
                    new_poly = _construct_from_supports(supports, best_local_coeffs)
                    new_val_sq = _square_count(new_poly)
                    delta = new_val_sq - current_val
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        current_val = new_val_sq
                        if current_val < best_val:
                            best_val = current_val
                            best_poly = new_poly
                    else:
                        best_local_coeffs[idx] = old_val
                    T *= 0.99
                    
                if best_val == 12:
                    break

    return best_poly
