"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_sq_nonzero(coeff_list):
    """Count nonzero terms in the square of polynomial given as coefficient list."""
    # Use FFT for fast convolution
    n = len(coeff_list)
    # Need convolution of size up to 2n-1
    size = 1
    while size < 2 * n - 1:
        size *= 2
    arr = np.zeros(size, dtype=np.float64)
    arr[:n] = coeff_list
    fft_arr = np.fft.fft(arr)
    sq_fft = fft_arr * fft_arr
    sq_arr = np.real(np.fft.ifft(sq_fft))
    # Round to nearest integer
    sq_arr = np.round(sq_arr).astype(np.int64)
    # The square has degree up to 2*(n-1), so length 2n-1
    sq_poly = sq_arr[:2*n-1]
    return int(np.sum(sq_poly != 0))

def _validate(coeff_list):
    """Check if coefficient list is valid. Returns True if valid."""
    if len(coeff_list) > 41:
        return False
    if len(coeff_list) < 1:
        return False
    if coeff_list[-1] == 0:
        return False
    nz = sum(1 for c in coeff_list if c != 0)
    if nz != 12:
        return False
    for c in coeff_list:
        if c < -64 or c > 64:
            return False
    return True

def _make_valid_candidate(exps, coefs):
    """Create a coefficient list from exponents and coefficients. Returns None if invalid."""
    if len(exps) != 12 or len(coefs) != 12:
        return None
    # Check exponents are valid
    if any(e < 0 or e >= 41 for e in exps):
        return None
    if len(set(exps)) != 12:
        return None
    # Check coefficients are nonzero and in range
    for c in coefs:
        if c == 0 or c < -64 or c > 64:
            return None
    max_deg = max(exps)
    coeff_list = [0] * (max_deg + 1)
    for i in range(12):
        coeff_list[exps[i]] = coefs[i]
    if not _validate(coeff_list):
        return None
    return coeff_list

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    best_coeffs = None
    best_sq = 999
    
    def _time_check():
        return time.time() > deadline
    
    # Phase 1: Quick brute-force on sign patterns with ±1 coefficients
    # This is fast and gives a good starting point
    candidate_exps_list = []
    for d in range(1, 5):
        for a in range(0, 41 - 11*d):
            exps = [a + i*d for i in range(12)]
            if max(exps) <= 40:
                candidate_exps_list.append(tuple(exps))
    # Add some random sets
    random.seed(42)
    for _ in range(100):
        exps = sorted(random.sample(range(0, 41), 12))
        candidate_exps_list.append(tuple(exps))
    candidate_exps_list = list(set(candidate_exps_list))
    
    for exps in candidate_exps_list:
        if _time_check():
            break
        # Brute force 2^12 sign patterns
        for mask in range(4096):
            if _time_check():
                break
            signs = []
            for i in range(12):
                signs.append(1 if (mask >> i) & 1 else -1)
            coeff_list = [0] * (max(exps) + 1)
            for i in range(12):
                coeff_list[exps[i]] = signs[i]
            if _validate(coeff_list):
                sq = _count_sq_nonzero(coeff_list)
                if sq < best_sq:
                    best_sq = sq
                    best_coeffs = list(coeff_list)
    
    if _time_check():
        if best_coeffs is not None:
            return best_coeffs
        return [1] * 12
    
    # Phase 2: Genetic algorithm with full coefficient values
    # Population of (exps, coefs) pairs
    pop_size = 200
    pop = []
    
    # Initialize population
    # Start with best found and random perturbations
    if best_coeffs is not None:
        # Extract exps and coefs from best
        best_exps = [i for i, c in enumerate(best_coeffs) if c != 0]
        best_coefs = [c for c in best_coeffs if c != 0]
        pop.append((list(best_exps), list(best_coefs)))
    
    # Add random individuals
    for _ in range(pop_size - 1):
        exps = sorted(random.sample(range(0, 41), 12))
        coefs = [random.choice([-1, 1]) for _ in range(12)]
        pop.append((exps, coefs))
    
    # Also add some with larger coefficients
    for _ in range(pop_size // 4):
        exps = sorted(random.sample(range(0, 41), 12))
        coefs = [random.randint(-64, 64) for _ in range(12)]
        # Ensure nonzero
        for i in range(12):
            while coefs[i] == 0:
                coefs[i] = random.randint(-64, 64)
        pop.append((exps, coefs))
    
    # Evaluate initial population
    pop_scores = []
    for exps, coefs in pop:
        cl = _make_valid_candidate(exps, coefs)
        if cl is None:
            pop_scores.append(999)
        else:
            sq = _count_sq_nonzero(cl)
            pop_scores.append(sq)
            if sq < best_sq:
                best_sq = sq
                best_coeffs = cl
    
    # Genetic algorithm loop
    ga_generations = 0
    while not _time_check() and ga_generations < 5000:
        ga_generations += 1
        
        # Sort by fitness
        indexed = sorted(range(len(pop)), key=lambda i: pop_scores[i])
        
        # Selection: keep top 50%
        elite_count = max(10, len(pop) // 2)
        new_pop = []
        for i in range(elite_count):
            idx = indexed[i]
            new_pop.append((list(pop[idx][0]), list(pop[idx][1])))
        
        # Crossover and mutation to fill rest
        while len(new_pop) < pop_size:
            if _time_check():
                break
            # Select two parents (tournament selection)
            p1_idx = random.choice(indexed[:elite_count])
            p2_idx = random.choice(indexed[:elite_count])
            p1_exps, p1_coefs = pop[p1_idx]
            p2_exps, p2_coefs = pop[p2_idx]
            
            # Crossover: mix exponents and coefficients
            child_exps = []
            child_coefs = []
            for i in range(12):
                if random.random() < 0.5:
                    child_exps.append(p1_exps[i])
                    child_coefs.append(p1_coefs[i])
                else:
                    child_exps.append(p2_exps[i])
                    child_coefs.append(p2_coefs[i])
            child_exps = sorted(child_exps)
            
            # Mutation
            if random.random() < 0.3:
                # Mutate an exponent
                idx = random.randint(0, 11)
                old_e = child_exps[idx]
                child_exps[idx] = random.randint(0, 40)
                child_exps = sorted(child_exps)
                # Check for duplicates
                if len(set(child_exps)) < 12:
                    # Revert
                    child_exps[idx] = old_e
                    child_exps = sorted(child_exps)
            
            if random.random() < 0.4:
                # Mutate a coefficient
                idx = random.randint(0, 11)
                child_coefs[idx] = random.randint(-64, 64)
                while child_coefs[idx] == 0:
                    child_coefs[idx] = random.randint(-64, 64)
            
            if random.random() < 0.1:
                # Large mutation: change multiple things
                for _ in range(random.randint(2, 5)):
                    if random.random() < 0.5:
                        idx = random.randint(0, 11)
                        child_exps[idx] = random.randint(0, 40)
                        child_exps = sorted(child_exps)
                    else:
                        idx = random.randint(0, 11)
                        child_coefs[idx] = random.randint(-64, 64)
                        while child_coefs[idx] == 0:
                            child_coefs[idx] = random.randint(-64, 64)
            
            # Validate
            cl = _make_valid_candidate(child_exps, child_coefs)
            if cl is not None:
                sq = _count_sq_nonzero(cl)
                if sq < best_sq:
                    best_sq = sq
                    best_coeffs = cl
                new_pop.append((child_exps, child_coefs))
            else:
                # Try to fix: remove duplicates
                unique_exps = list(dict.fromkeys(child_exps))
                if len(unique_exps) == 12:
                    cl = _make_valid_candidate(unique_exps, child_coefs)
                    if cl is not None:
                        sq = _count_sq_nonzero(cl)
                        if sq < best_sq:
                            best_sq = sq
                            best_coeffs = cl
                        new_pop.append((unique_exps, child_coefs))
        
        pop = new_pop[:pop_size]
        
        # Re-evaluate if needed (scores should be carried over for elites)
        # For simplicity, re-evaluate all
        pop_scores = []
        for exps, coefs in pop:
            cl = _make_valid_candidate(exps, coefs)
            if cl is None:
                pop_scores.append(999)
            else:
                sq = _count_sq_nonzero(cl)
                pop_scores.append(sq)
                if sq < best_sq:
                    best_sq = sq
                    best_coeffs = cl
    
    if best_coeffs is not None:
        return best_coeffs
    
    # Fallback
    return [1] * 12
# EVOLVE-BLOCK-END
