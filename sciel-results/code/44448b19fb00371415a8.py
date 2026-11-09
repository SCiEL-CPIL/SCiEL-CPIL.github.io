import time
import random

def _count_nonzero_square(coeffs, exponents):
    # coeffs and exponents are lists of same length
    # Create a dict of degree -> coeff for the polynomial
    poly = {}
    for c, e in zip(coeffs, exponents):
        if c != 0:
            poly[e] = poly.get(e, 0) + c
    
    # Calculate square
    square = {}
    keys = list(poly.keys())
    for i in range(len(keys)):
        e1, c1 = keys[i], poly[keys[i]]
        for j in range(i, len(keys)):
            e2, c2 = keys[j], poly[keys[j]]
            e_sum = e1 + e2
            c_prod = c1 * c2
            if i == j:
                square[e_sum] = square.get(e_sum, 0) + c_prod
            else:
                square[e_sum] = square.get(e_sum, 0) + 2 * c_prod
    
    # Count non-zeros
    count = 0
    for v in square.values():
        if v != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Initial candidate based on Q(x)(1+x^5)
    # Q(x) = 1+x+x^2-x^3-x^4-x^5
    # P(x) = Q(x)(1+x^5) = 1+x+x^2-x^3-x^4-x^5 + x^5+x^6+x^7-x^8-x^9-x^10
    # P(x) = 1+x+x^2-x^3-x^4+0x^5+x^6+x^7-x^8-x^9-x^10
    # Wait, the term x^5 cancels?
    # Q(x) = 1+x+x^2-x^3-x^4-x^5.
    # P(x) = Q(x) + x^5 Q(x) = 1+x+x^2-x^3-x^4-x^5 + x^5+x^6+x^7-x^8-x^9-x^10
    # P(x) = 1+x+x^2-x^3-x^4+x^6+x^7-x^8-x^9-x^10.
    # This has 10 terms! We need 12.
    # So we should pick a Q with 6 terms that doesn't cancel with the shift.
    # Or just use a random 12-term polynomial.
    
    # Let's use a random start and refine.
    best_coeffs = []
    best_exponents = []
    best_score = float('inf')
    
    def get_score(coeffs, exponents):
        # Check validity
        if len(set(exponents)) != len(exponents):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if any(c < -64 or c > 64 for c in coeffs):
            return float('inf')
        if any(e < 0 or e > 40 for e in exponents):
            return float('inf')
        
        score = _count_nonzero_square(coeffs, exponents)
        return score

    # Try a few random starts
    for _ in range(5):
        exponents = random.sample(range(41), 12)
        exponents.sort()
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        
        score = get_score(coeffs, exponents)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs[:]
            best_exponents = exponents[:]
            
        # Local search
        current_coeffs = coeffs[:]
        current_exponents = exponents[:]
        current_score = score
        
        no_improvement = 0
        while time.time() < deadline and no_improvement < 200:
            improved = False
            # Randomly pick a move type
            move_type = random.choice(['coeff', 'exp', 'swap'])
            
            idx = random.randint(0, 11)
            
            if move_type == 'coeff':
                old_c = current_coeffs[idx]
                new_c = old_c + random.choice([-1, 1])
                if -64 <= new_c <= 64 and new_c != 0:
                    current_coeffs[idx] = new_c
                    new_score = get_score(current_coeffs, current_exponents)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_coeffs = current_coeffs[:]
                            best_exponents = current_exponents[:]
                    else:
                        current_coeffs[idx] = old_c
                        no_improvement += 1
                else:
                    no_improvement += 1
            elif move_type == 'exp':
                old_e = current_exponents[idx]
                new_e = old_e + random.choice([-1, 1])
                if 0 <= new_e <= 40 and new_e not in current_exponents:
                    current_exponents[idx] = new_e
                    current_exponents.sort()
                    new_score = get_score(current_coeffs, current_exponents)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_coeffs = current_coeffs[:]
                            best_exponents = current_exponents[:]
                    else:
                        current_exponents[idx] = old_e
                        current_exponents.sort()
                        no_improvement += 1
                else:
                    no_improvement += 1
            elif move_type == 'swap':
                j = random.randint(0, 11)
                if i != j:
                    current_exponents[i], current_exponents[j] = current_exponents[j], current_exponents[i]
                    current_exponents.sort()
                    new_score = get_score(current_coeffs, current_exponents)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_coeffs = current_coeffs[:]
                            best_exponents = current_exponents[:]
                    else:
                        current_exponents[i], current_exponents[j] = current_exponents[j], current_exponents[i]
                        current_exponents.sort()
                        no_improvement += 1
                else:
                    no_improvement += 1
            
            if improved:
                no_improvement = 0
                
    # Construct final list
    # Sort by exponent
    pairs = sorted(zip(best_exponents, best_coeffs))
    max_deg = max(p[0] for p in pairs)
    result = [0] * (max_deg + 1)
    for e, c in pairs:
        result[e] = c
        
    # Ensure length <= 41
    if len(result) > 41:
        # This shouldn't happen if exponents are < 41
        result = result[:41]
        
    # Ensure nonzero final coefficient
    # The loop ensures max_deg is the highest exponent, so result[-1] is nonzero.
    
    return result

# EVOLVE-BLOCK-END
