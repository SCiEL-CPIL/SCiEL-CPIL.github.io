import random
import time

def count_nonzero_square(terms):
    """
    terms: list of (exp, coeff)
    Returns number of non-zero coefficients in the square.
    """
    sq = {}
    n = len(terms)
    for i in range(n):
        e1, c1 = terms[i]
        for j in range(i, n):
            e2, c2 = terms[j]
            e_sum = e1 + e2
            val = c1 * c2
            if i == j:
                sq[e_sum] = sq.get(e_sum, 0) + val
            else:
                sq[e_sum] = sq.get(e_sum, 0) + 2 * val
    return sum(1 for v in sq.values() if v != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    best_score = 0.0
    best_terms = None
    
    # Helper to create a random valid polynomial
    def random_poly():
        exps = random.sample(range(0, 41), 12)
        exps.sort()
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        return list(zip(exps, coeffs))

    # Run multiple restarts
    while time.time() < deadline:
        terms = random_poly()
        curr_nz = count_nonzero_square(terms)
        curr_score = 12.0 / curr_nz
        
        local_best_terms = terms[:]
        local_best_score = curr_score
        local_best_nz = curr_nz
        
        # Local search phase
        no_improve = 0
        T = 1.0 # Temperature for simulated annealing (simple version)
        
        while time.time() < deadline and no_improve < 100:
            # Generate neighbor
            new_terms = local_best_terms[:]
            idx = random.randint(0, 11)
            e, c = new_terms[idx]
            
            if random.random() < 0.8:
                # Perturb exponent
                de = random.choice([-2, -1, 1, 2])
                new_e = e + de
                if 0 <= new_e <= 40:
                    # Check uniqueness
                    exps = [t[0] for t in new_terms]
                    exps[idx] = new_e
                    if len(set(exps)) == 12:
                        new_terms[idx] = (new_e, c)
                    else:
                        no_improve += 1
                        continue
            else:
                # Perturb coefficient
                # Try small integers
                new_c = random.choice([-2, -1, 1, 2])
                if new_c != c:
                    new_terms[idx] = (e, new_c)
                else:
                    no_improve += 1
                    continue
            
            new_nz = count_nonzero_square(new_terms)
            new_score = 12.0 / new_nz
            
            if new_score > local_best_score:
                local_best_terms = new_terms
                local_best_score = new_score
                local_best_nz = new_nz
                no_improve = 0
            else:
                # Accept worse with some probability to escape local minima
                # Delta is negative
                delta = new_score - local_best_score
                # Simple acceptance: if random < 0.1 and not too bad
                if random.random() < 0.1 and new_nz <= local_best_nz + 2:
                    local_best_terms = new_terms
                    local_best_score = new_score
                    local_best_nz = new_nz
                else:
                    no_improve += 1
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_terms = local_best_terms

    if best_terms is None:
        return [1] * 12

    max_exp = max(t[0] for t in best_terms)
    res = [0] * (max_exp + 1)
    for e, c in best_terms:
        res[e] = c
    return res
