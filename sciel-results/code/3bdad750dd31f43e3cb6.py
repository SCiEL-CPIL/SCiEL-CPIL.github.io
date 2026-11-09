import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n_terms = 12
    max_degree = 40
    
    best_score = 100
    best_P = None
    
    # Precompute pairs (i, j) with i <= j
    pairs = []
    for i in range(n_terms):
        for j in range(i, n_terms):
            pairs.append((i, j))
            
    def evaluate(exponents, signs):
        # exponents: list of 12
        # signs: list of 12
        coeff_dict = {}
        
        for i, j in pairs:
            d = exponents[i] + exponents[j]
            if i == j:
                val = 1 
            else:
                val = 2 * signs[i] * signs[j]
            
            if d in coeff_dict:
                coeff_dict[d] += val
            else:
                coeff_dict[d] = val
        
        score = 0
        for v in coeff_dict.values():
            if v != 0:
                score += 1
        return score

    def make_P(exponents, signs):
        if len(set(exponents)) != n_terms:
            return None
        if min(exponents) < 0 or max(exponents) > max_degree:
            return None
        
        P = [0] * (max_degree + 1)
        for i in range(n_terms):
            P[exponents[i]] = signs[i]
            
        last_nonzero = -1
        for i in range(len(P)-1, -1, -1):
            if P[i] != 0:
                last_nonzero = i
                break
        if last_nonzero == -1:
            return None
        return P[:last_nonzero+1]

    # Initialize with diverse starts
    starts = []
    starts.append((list(range(12)), [1]*12))
    starts.append((list(range(12)), [1 if i%2==0 else -1 for i in range(12)]))
    
    # Add some structured starts
    # Geometric-like
    exps = [0, 1, 3, 7, 15, 31, 32, 33, 34, 35, 36, 37]
    if max(exps) <= max_degree:
        starts.append((exps, [1]*12))
        
    for _ in range(20):
        exp = sorted(random.sample(range(0, max_degree+1), n_terms))
        sig = [random.choice([-1, 1]) for _ in range(n_terms)]
        starts.append((exp, sig))

    # Evaluate starts
    current_exp = starts[0][0][:]
    current_sig = starts[0][1][:]
    current_score = evaluate(current_exp, current_sig)
    
    for exp, sig in starts:
        score = evaluate(exp, sig)
        if score < current_score:
            current_exp = exp[:]
            current_sig = sig[:]
            current_score = score
            
        if score < best_score:
            best_score = score
            P = make_P(exp, sig)
            if P:
                best_P = P

    # Search Loop
    while time.time() < deadline:
        new_exp = current_exp[:]
        new_sig = current_sig[:]
        
        r = random.random()
        if r < 0.6:
            # Perturb one exponent
            idx = random.randint(0, n_terms-1)
            delta = random.choice([-2, -1, 1, 2, -3, 3, -4, 4])
            new_exp[idx] += delta
        elif r < 0.85:
            # Flip one sign
            idx = random.randint(0, n_terms-1)
            new_sig[idx] *= -1
        elif r < 0.95:
            # Perturb two exponents
            idx1 = random.randint(0, n_terms-1)
            idx2 = random.randint(0, n_terms-1)
            if idx1 != idx2:
                new_exp[idx1] += random.choice([-1, 1])
                new_exp[idx2] += random.choice([-1, 1])
        else:
            # Random jump
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, n_terms-1)
                new_exp[idx] = random.randint(0, max_degree)
            
        # Validate
        if len(set(new_exp)) != n_terms or min(new_exp) < 0 or max(new_exp) > max_degree:
            continue
            
        new_score = evaluate(new_exp, new_sig)
        
        # Acceptance
        if new_score < current_score:
            current_exp = new_exp
            current_sig = new_sig
            current_score = new_score
            
            P = make_P(current_exp, current_sig)
            if P and current_score < best_score:
                best_score = current_score
                best_P = P
        elif random.random() < 0.05:
            # Escape local minima
            current_exp = new_exp
            current_sig = new_sig
            current_score = new_score

    if best_P is None:
        return [1] * 12
        
    return best_P
