# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _square_count(coeffs_list):
    """
    Calculate the number of non-zero terms in the square of a polynomial
    represented as a list of coefficients (index = degree).
    """
    if not coeffs_list:
        return 0
    # Use numpy convolve for speed
    sq = np.convolve(coeffs_list, coeffs_list)
    return np.count_nonzero(sq)

def _construct_from_terms(terms):
    """
    Construct a polynomial list from a list of (exp, coeff) pairs.
    """
    if not terms:
        return []
    max_exp = max(e for e, c in terms)
    poly = [0] * (max_exp + 1)
    for e, c in terms:
        if 0 <= e < len(poly):
            poly[e] += c
    return poly

def _evaluate_P_from_Q(Q_terms):
    """
    Given Q terms (list of (exp, coeff)), compute P = (1+x)Q.
    Returns P list or None if invalid.
    """
    # P = Q + xQ
    # Term (e, c) in Q contributes to P at e (c) and e+1 (c).
    # We need exactly 12 non-zero terms in P.
    # This requires that no two contributions to the same exponent in P cancel out to 0
    # AND that the total number of non-zero entries is exactly 12.
    
    # Map exponent -> coeff sum for P
    p_coeffs = {}
    for e, c in Q_terms:
        p_coeffs[e] = p_coeffs.get(e, 0) + c
        p_coeffs[e+1] = p_coeffs.get(e+1, 0) + c
    
    # Filter non-zeros
    p_nonzero = {e: c for e, c in p_coeffs.items() if c != 0}
    
    if len(p_nonzero) != 12:
        return None
        
    # Check bounds
    if any(abs(c) > 64 for c in p_nonzero.values()):
        return None
        
    max_exp = max(p_nonzero.keys())
    if max_exp >= 41:
        return None
        
    poly = [0] * (max_exp + 1)
    for e, c in p_nonzero.items():
        poly[e] = c
        
    if poly[-1] == 0:
        return None
        
    return poly

def _search_Q_strategy(deadline):
    """
    Search for Q (11 terms) such that P=(1+x)Q has sparse square.
    """
    best_score = 1000
    best_poly = None
    
    # Initial candidates for Q
    # Q has 11 terms.
    # Exponents should be within 0..39 so P fits in 0..40.
    # Coefficients small: -2, -1, 1, 2
    
    candidates = []
    
    # Generate random Qs
    for _ in range(100):
        exps = sorted(random.sample(range(0, 40), 11))
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(11)]
        # Ensure no adjacent opposite signs to keep P at 12 terms?
        # Actually, _evaluate_P_from_Q checks the count.
        q_terms = list(zip(exps, coeffs))
        candidates.append(q_terms)
        
    # Local search on Q terms
    for q_terms in candidates:
        if time.time() > deadline - 2.0:
            break
            
        P = _evaluate_P_from_Q(q_terms)
        if P is None:
            continue
            
        score = _square_count(P)
        if score < best_score:
            best_score = score
            best_poly = P[:]
            
        # Refine Q
        # Moves: change coeff, change exp, swap exps
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
                
            move_type = random.random()
            new_q_terms = [t[:] for t in q_terms]
            
            if move_type < 0.5:
                # Change coeff
                idx = random.randint(0, 10)
                old_c = new_q_terms[idx][1]
                new_c = random.choice([-2, -1, 1, 2])
                if new_c == old_c: continue
                new_q_terms[idx][1] = new_c
            elif move_type < 0.8:
                # Change exp
                idx = random.randint(0, 10)
                old_e = new_q_terms[idx][0]
                # Shift by 1 or 2
                delta = random.choice([-2, -1, 1, 2])
                new_e = old_e + delta
                if new_e < 0 or new_e > 39: continue
                # Check collision with other exps?
                # Q must have distinct exps?
                # If exps collide, Q has fewer than 11 terms.
                # We need Q to have 11 terms.
                if any(new_q_terms[j][0] == new_e for j in range(11) if j != idx):
                    continue
                new_q_terms[idx][0] = new_e
            else:
                # Swap exps
                i, j = random.sample(range(11), 2)
                new_q_terms[i][0], new_q_terms[j][0] = new_q_terms[j][0], new_q_terms[i][0]
                # Check bounds and distinctness
                if any(e < 0 or e > 39 for e, c in new_q_terms): continue
                if len(set(e for e, c in new_q_terms)) != 11: continue
                
            new_q_terms.sort(key=lambda t: t[0])
            P_new = _evaluate_P_from_Q(new_q_terms)
            if P_new is None:
                continue
                
            score_new = _square_count(P_new)
            
            # Accept if better
            if score_new < score:
                score = score_new
                q_terms = new_q_terms
                if score < best_score:
                    best_score = score
                    best_poly = P_new[:]
                    
    return best_poly, best_score

def _search_AB_strategy(deadline):
    """
    Search for A (6 terms) and B (6 terms) such that P=AB has sparse square.
    P^2 = A^2 B^2.
    """
    best_score = 1000
    best_poly = None
    
    # Generate random A and B
    # A, B have 6 terms.
    # Exponents 0..20 roughly.
    
    for _ in range(50):
        if time.time() > deadline - 2.0:
            break
            
        exps_A = sorted(random.sample(range(0, 21), 6))
        coeffs_A = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        A_terms = list(zip(exps_A, coeffs_A))
        
        exps_B = sorted(random.sample(range(0, 21), 6))
        coeffs_B = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
        B_terms = list(zip(exps_B, coeffs_B))
        
        # Construct A and B polynomials
        A_poly = _construct_from_terms(A_terms)
        B_poly = _construct_from_terms(B_terms)
        
        if len(A_poly) == 0 or len(B_poly) == 0: continue
        if any(abs(c) > 64 for c in A_poly) or any(abs(c) > 64 for c in B_poly): continue
        
        # P = A * B
        P_poly = np.convolve(A_poly, B_poly)
        P_poly = P_poly.astype(int)
        
        if len(P_poly) > 41: continue
        if P_poly[-1] == 0: continue
        
        # Check exactly 12 non-zero terms in P
        if np.count_nonzero(P_poly) != 12: continue
        
        score = _square_count(P_poly)
        if score < best_score:
            best_score = score
            best_poly = P_poly.tolist()
            
        # Local search on A and B?
        # Too complex for simple implementation.
        # Just do more random restarts.
        
    return best_poly, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_poly = None
    best_score = 1000
    
    # Strategy 1: P = (1+x)Q
    poly1, score1 = _search_Q_strategy(deadline)
    if poly1 and score1 < best_score:
        best_score = score1
        best_poly = poly1
        
    # Strategy 2: P = AB
    if time.time() < deadline - 5.0:
        poly2, score2 = _search_AB_strategy(deadline)
        if poly2 and score2 < best_score:
            best_score = score2
            best_poly = poly2
            
    # Fallback: Original random search if needed
    if best_poly is None or best_score > 20:
         # Quick random search for safety
         for _ in range(100):
             exps = sorted(random.sample(range(0, 41), 12))
             signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
             coeffs = [1 if random.random() > 0.5 else 2 for _ in range(12)]
             poly = [0] * 41
             for i in range(12):
                 poly[exps[i]] = signs[i] * coeffs[i]
             if poly[-1] != 0 and np.count_nonzero(poly) == 12:
                 score = _square_count(poly)
                 if score < best_score:
                     best_score = score
                     best_poly = poly[:]
                     
    if best_poly is None:
        return [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        
    return best_poly

# EVOLVE-BLOCK-END
