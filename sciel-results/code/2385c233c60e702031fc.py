import time
import random
from collections import defaultdict

def _square_nonzero_count(exponents, coefficients):
    n = len(exponents)
    square = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            e = exponents[i] + exponents[j]
            c = coefficients[i] * coefficients[j]
            if i != j:
                c *= 2
            square[e] += c
    return sum(1 for c in square.values() if c != 0)

def _build_coeff_list(exponents, coefficients, max_degree=40):
    max_exp = max(exponents)
    if max_exp > max_degree:
        return None
    result = [0] * (max_exp + 1)
    for e, c in zip(exponents, coefficients):
        result[e] = c
    nonzero = sum(1 for x in result if x != 0)
    if nonzero != 12:
        return None
    if result[-1] == 0:
        return None
    if any(abs(x) > 64 for x in result):
        return None
    if len(result) > 41:
        return None
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    fallback = [1] * 12
    fallback_score = 21
    best_score = fallback_score
    best_candidate = fallback

    def try_candidate(exps, coeffs):
        nonlocal best_score, best_candidate
        score = _square_nonzero_count(exps, coeffs)
        if score < best_score:
            candidate = _build_coeff_list(exps, coeffs)
            if candidate is not None:
                best_score = score
                best_candidate = candidate
                return True
        return False

    # Strategy 1: Factorization-based construction
    # P = A * B, where A and B are polynomials with few terms.
    # If A has k terms and B has m terms, P has at most k*m terms.
    # We want P to have exactly 12 terms.
    # Possible (k, m) pairs: (2,6), (3,4), (4,3), (6,2), (1,12), (12,1).
    # Let's focus on (3,4) and (2,6) as they balance term count and collision potential.
    
    def generate_factorization_candidates():
        candidates = []
        
        # Case 3x4
        # A: 3 terms, B: 4 terms
        for a_start in range(0, 5):
            for a_step in range(1, 5):
                a_exps = [a_start + i*a_step for i in range(3)]
                if max(a_exps) > 20: continue
                for b_start in range(0, 15):
                    for b_step in range(1, 5):
                        b_exps = [b_start + i*b_step for i in range(4)]
                        if max(b_exps) > 25: continue
                        
                        # Multiply A and B
                        p_exps = []
                        p_coeffs = []
                        for i in range(3):
                            for j in range(4):
                                p_exps.append(a_exps[i] + b_exps[j])
                                p_coeffs.append(1) # Initial coeff, will be optimized
            
            # Deduplicate exponents? No, if exponents collide, we sum coefficients.
            # But we need exactly 12 distinct exponents in the final P.
            # So we require that all 12 sums a_i + b_j are distinct.
            if len(set(p_exps)) == 12 and max(p_exps) <= 40:
                # Sort by exponent to align coefficients
                pairs = sorted(zip(p_exps, p_coeffs))
                exps = [x[0] for x in pairs]
                coeffs = [x[1] for x in pairs]
                candidates.append((exps, coeffs))
        
        # Case 2x6
        for a_start in range(0, 5):
            for a_step in range(1, 5):
                a_exps = [a_start + i*a_step for i in range(2)]
                if max(a_exps) > 20: continue
                for b_start in range(0, 20):
                    for b_step in range(1, 5):
                        b_exps = [b_start + i*b_step for i in range(6)]
                        if max(b_exps) > 30: continue
                        
                        p_exps = []
                        p_coeffs = []
                        for i in range(2):
                            for j in range(6):
                                p_exps.append(a_exps[i] + b_exps[j])
                                p_coeffs.append(1)
            
            if len(set(p_exps)) == 12 and max(p_exps) <= 40:
                pairs = sorted(zip(p_exps, p_coeffs))
                exps = [x[0] for x in pairs]
                coeffs = [x[1] for x in pairs]
                candidates.append((exps, coeffs))
                
        return candidates

    # Strategy 2: Simulated Annealing
    # State: (exponents, coefficients)
    # Exponents: sorted list of 12 unique integers in [0, 40]
    # Coefficients: list of 12 integers in [-64, 64]
    
    def anneal(exps_init, coeffs_init, max_steps=5000):
        exps = list(exps_init)
        coeffs = list(coeffs_init)
        current_score = _square_nonzero_count(exps, coeffs)
        best_local_score = current_score
        best_local_exps = exps[:]
        best_local_coeffs = coeffs[:]
        
        T = 10.0
        cooling = 0.999
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if T < 0.01:
                break
                
            # Perturb
            if random.random() < 0.5:
                # Change one coefficient
                i = random.randint(0, 11)
                old_c = coeffs[i]
                new_c = random.randint(-4, 4)
                if new_c == 0: new_c = 1
                coeffs[i] = new_c
                new_score = _square_nonzero_count(exps, coeffs)
                delta = new_score - current_score
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    current_score = new_score
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_exps = exps[:]
                        best_local_coeffs = coeffs[:]
                else:
                    coeffs[i] = old_c
            else:
                # Change one exponent (shift by 1 or swap)
                i = random.randint(0, 11)
                old_e = exps[i]
                # Try shifting
                shift = random.choice([-1, 1, -2, 2])
                new_e = old_e + shift
                if 0 <= new_e <= 40 and new_e not in exps:
                    exps[i] = new_e
                    exps.sort()
                    # Re-align coeffs? No, coeffs are tied to the *term*, not the index.
                    # Wait, my representation is (exps, coeffs) where coeffs[i] corresponds to exps[i].
                    # If I change exps[i], I must keep coeffs[i] with it.
                    # But if I sort exps, I must sort coeffs accordingly.
                    # This is tricky. Let's keep a list of (e, c) pairs.
                    # Actually, let's stick to the list representation but be careful.
                    # If I swap exps, I swap coeffs.
                    # If I shift exps, I keep coeffs.
                    
                    new_score = _square_nonzero_count(exps, coeffs)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_score = new_score
                        if current_score < best_local_score:
                            best_local_score = current_score
                            best_local_exps = exps[:]
                            best_local_coeffs = coeffs[:]
                    else:
                        exps[i] = old_e
                        exps.sort()
                # Try swapping two exponents
                if random.random() < 0.5:
                    j = random.randint(0, 11)
                    if i != j:
                        exps[i], exps[j] = exps[j], exps[i]
                        coeffs[i], coeffs[j] = coeffs[j], coeffs[i]
                        exps.sort()
                        # Re-sort coeffs to match exps? No, swapping changes which coeff goes with which exp.
                        # This is getting complicated with the sorted list constraint.
                        # Let's abandon the sorted list constraint during annealing and only sort at the end.
                        # But _square_nonzero_count doesn't care about order.
                        # So we can keep unsorted lists during annealing.
                        # However, _build_coeff_list needs to map exp->coeff.
                        # So we just need unique exponents.
                        
                        new_score = _square_nonzero_count(exps, coeffs)
                        delta = new_score - current_score
                        if delta < 0 or random.random() < pow(2.718, -delta / T):
                            current_score = new_score
                            if current_score < best_local_score:
                                best_local_score = current_score
                                best_local_exps = exps[:]
                                best_local_coeffs = coeffs[:]
                        else:
                            exps[i], exps[j] = exps[j], exps[i]
                            coeffs[i], coeffs[j] = coeffs[j], coeffs[i]
            
            T *= cooling
            
        return best_local_score, best_local_exps, best_local_coeffs

    # Initialize candidates
    init_candidates = generate_factorization_candidates()
    
    # Also add some random structured starts
    for _ in range(50):
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        init_candidates.append((exps, coeffs))
        
    # Run Annealing on top candidates
    # Sort initial candidates by score
    scored_inits = []
    for exps, coeffs in init_candidates:
        score = _square_nonzero_count(exps, coeffs)
        scored_inits.append((score, exps, coeffs))
    scored_inits.sort(key=lambda x: x[0])
    
    # Take top 20 for annealing
    for score, exps, coeffs in scored_inits[:20]:
        if time.time() > deadline - 10:
            break
        best_local_score, best_local_exps, best_local_coeffs = anneal(exps, coeffs, max_steps=2000)
        try_candidate(best_local_exps, best_local_coeffs)
        
    # If we found a good one, return it
    if best_score < fallback_score:
        return best_candidate
        
    # Fallback to a simple heuristic if SA didn't improve
    # Try the pure AP with alternating signs
    for start in range(0, 5):
        for step in range(1, 5):
            if time.time() > deadline:
                break
            exps = [start + i*step for i in range(12)]
            if max(exps) > 40: continue
            coeffs = [1 if i%2==0 else -1 for i in range(12)]
            try_candidate(exps, coeffs)
            
    return best_candidate

# EVOLVE-BLOCK-END
