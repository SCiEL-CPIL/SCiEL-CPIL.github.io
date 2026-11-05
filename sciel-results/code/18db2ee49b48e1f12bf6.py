import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_poly = None
    best_score = float('inf')
    
    def count_square_terms(d):
        """Count nonzero terms in the exact integer square of a sparse polynomial."""
        if not d:
            return 0
        square = defaultdict(int)
        items = list(d.items())
        n = len(items)
        for i in range(n):
            ei, ci = items[i]
            for j in range(i, n):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    square[key] += ci * ci
                else:
                    square[key] += 2 * ci * cj
        return sum(1 for v in square.values() if v != 0)

    def validate_and_score(d):
        """Validate constraints and return score if valid, else None."""
        if not d:
            return None
        max_exp = max(d.keys())
        if max_exp > 40:
            return None
        n = max_exp + 1
        coeffs_list = [0] * n
        for e, c in d.items():
            coeffs_list[e] = c
        # Trim trailing zeros
        while len(coeffs_list) > 1 and coeffs_list[-1] == 0:
            coeffs_list.pop()
        if len(coeffs_list) > 41:
            return None
        if coeffs_list[-1] == 0:
            return None
        nonzero_count = sum(1 for c in coeffs_list if c != 0)
        if nonzero_count != 12:
            return None
        for c in coeffs_list:
            if not (-64 <= c <= 64):
                return None
        return count_square_terms(d)

    def dict_to_list(d):
        if not d:
            return [1]
        max_exp = max(d.keys())
        lst = [0] * (max_exp + 1)
        for e, c in d.items():
            lst[e] = c
        return lst

    # ====================================================================
    # STRATEGY 1: Exhaustive search over all 2^12 sign patterns for consecutive exponents.
    # This is fast (4096 iterations) and guarantees the best result in the 
    # +/-1 coefficient space for the 0..11 exponent structure.
    # ====================================================================
    exps_consec = list(range(12))
    for signs in itertools.product([-1, 1], repeat=12):
        if time.time() > deadline:
            break
        d = {exps_consec[i]: signs[i] for i in range(12)}
        score = validate_and_score(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)

    # ====================================================================
    # STRATEGY 2: Structured "Block" Search with Larger Coefficients
    # We use the insight that cancellation in S_k = sum_{i+j=k} c_i c_j 
    # can be enhanced by using larger magnitudes that sum to zero.
    # Example: {1, 1, -2} sums to 0.
    # We partition the 12 terms into blocks and assign patterns that 
    # create internal cancellation within the block's contribution to the square.
    #
    # Pattern A: Pairs of (a, -a) at exponents e, e+1.
    #   (a x^e - a x^{e+1})^2 = a^2 x^{2e} - 2a^2 x^{2e+1} + a^2 x^{2e+2}
    #   Cross terms with other blocks might cancel if aligned.
    #
    # Pattern B: Triples (1, 1, -2) at exponents e, e+1, e+2.
    #   (x^e + x^{e+1} - 2x^{e+2}) ... complex.
    #
    # Instead, let's do a targeted search on "symmetric" coefficient sets.
    # If P(x) is palindromic or anti-palindromic, P^2 has symmetry.
    # 
    # Let's try: Coefficients from {-2, -1, 1, 2} with a heuristic that 
    # favors "balanced" sums.
    # ====================================================================
    
    # Use a faster exact counter for the inner loop
    def fast_count(d):
        if not d:
            return 0
        square = {}
        items = list(d.items())
        for i in range(len(items)):
            ei, ci = items[i]
            for j in range(i, len(items)):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    square[key] = square.get(key, 0) + ci * ci
                else:
                    square[key] = square.get(key, 0) + 2 * ci * cj
        return sum(1 for v in square.values() if v != 0)

    # Randomized search with "restarts" and "cooling" but focused on 
    # coefficient magnitudes that allow cancellation.
    # Key idea: If c_i + c_j = 0, then the term 2 c_i c_j x^{e_i+e_j} is negative 
    # and might cancel a positive term from c_k^2 or c_l c_m.
    
    random.seed(123)
    
    # Search space: Exponents are consecutive 0..11 (most dense, most cancellation potential).
    # Coefficients: Drawn from {-3, -2, -1, 1, 2, 3}
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Initialize with random small integers
        curr_coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        curr_d = {i: curr_coeffs[i] for i in range(12)}
        curr_score = fast_count(curr_d)
        
        if curr_score < best_score:
            best_score = curr_score
            best_poly = dict_to_list(curr_d)
        
        T = 5.0
        for step in range(3000):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 11)
            old_val = curr_coeffs[idx]
            new_val = random.choice([-3, -2, -1, 1, 2, 3])
            if new_val == old_val:
                continue
            
            curr_coeffs[idx] = new_val
            curr_d[idx] = new_val
            
            new_score = fast_count(curr_d)
            
            if new_score < curr_score:
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = dict_to_list(curr_d)
            else:
                delta = new_score - curr_score
                if random.random() < 2.718 ** (-delta / max(T, 0.01)):
                    curr_score = new_score
                else:
                    curr_coeffs[idx] = old_val
                    curr_d[idx] = old_val
            
            T *= 0.99

    # ====================================================================
    # STRATEGY 3: "Zero-Sum" Pattern Injection
    # Specifically search for patterns where the sum of coefficients in 
    # arithmetic progressions is zero.
    # e.g., c_0 + c_2 + c_4 ... 
    # This is hard to target directly, so we use a different structure:
    # P(x) = A(x)^2 - B(x)^2 ? No, P must have 12 terms.
    # 
    # Let's try: P(x) = (1 + x + ... + x^k) * Q(x)
    # If Q has few terms, P has (k+1)*terms(Q) terms.
    # If k=1, P = (1+x)Q. If Q has 6 terms, P has up to 12 terms.
    # P^2 = (1+x)^2 Q^2 = (1 + 2x + x^2) Q^2.
    # This structure might have fewer terms if Q^2 has structure.
    #
    # Let's search over Q with 6 terms, coefficients +/-1, exponents 0..5.
    # P = (1+x)Q.
    # ====================================================================
    
    # Generate all 2^6 patterns for Q
    for q_signs in itertools.product([-1, 1], repeat=6):
        if time.time() > deadline:
            break
        # Q has terms at 0..5
        q_d = {i: q_signs[i] for i in range(6)}
        # P = (1+x)Q => p_i = q_i + q_{i-1}
        p_d = {}
        for i in range(7):
            val = 0
            if i in q_d:
                val += q_d[i]
            if i-1 in q_d:
                val += q_d[i-1]
            if val != 0:
                p_d[i] = val
        
        # P should have 12 terms. (1+x)Q with 6 terms in Q usually has 7 or 8 terms in P.
        # This won't give 12 terms. We need Q to have more terms or different multiplier.
        # Try P = (1 + x^k) Q.
        # If Q has 6 terms, P has 12 terms if no overlaps.
        # Let's try P = (1 + x^10) Q where Q has 6 terms at 0..5.
        # P terms at 0..5 and 10..15. Total 12 terms.
        
        p_d2 = {}
        for i, c in q_d.items():
            p_d2[i] = c
            p_d2[i+10] = c
        
        if len(p_d2) != 12:
            continue
            
        score = validate_and_score(p_d2)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(p_d2)

    # Try P = (1 + x^5) Q where Q has 6 terms at 0..4? No, 5*2=10 terms.
    # Try P = (1 + x + x^2) Q where Q has 4 terms? 3*4=12 terms.
    # Q exponents 0..3.
    for q_signs in itertools.product([-1, 1], repeat=4):
        if time.time() > deadline:
            break
        q_d = {i: q_signs[i] for i in range(4)}
        p_d = {}
        for i, c in q_d.items():
            for j in range(3): # 1, x, x^2
                e = i + j
                p_d[e] = p_d.get(e, 0) + c
        
        # Check if exactly 12 nonzero terms
        if len([v for v in p_d.values() if v != 0]) != 12:
            continue
        
        # Clean zeros
        p_d_clean = {e: v for e, v in p_d.items() if v != 0}
        
        score = validate_and_score(p_d_clean)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(p_d_clean)

    # ====================================================================
    # STRATEGY 4: Final Fallback & Validation
    # ====================================================================
    if best_poly is None:
        # Fallback: 12 ones
        best_poly = [1] * 12
    
    # Ensure return format is correct
    # The evaluator expects a list.
    # Let's do a final sanity check.
    final_d = {i: c for i, c in enumerate(best_poly) if c != 0}
    if len(final_d) != 12:
        # This should not happen if our search is correct, but just in case
        best_poly = [1] * 12

    return best_poly
