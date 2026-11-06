import time
import random
import numpy as np
from itertools import combinations

def _square_nonzero_count(poly):
    """
    Given poly as a list of (exponent, coeff) pairs,
    return the number of nonzero terms in poly^2.
    """
    if not poly:
        return 0
    max_deg = max(exp for exp, _ in poly)
    sq_deg = 2 * max_deg
    # Build dense coefficient vector
    vec = np.zeros(sq_deg + 1, dtype=np.int64)
    for exp, coeff in poly:
        vec[exp] = coeff
    # Convolve
    result = np.convolve(vec, vec)
    return int(np.count_nonzero(result))

def _square_nonzero_count_fast(exps, coeffs):
    """
    Faster evaluation using sparse representation.
    exps: sorted list of exponents
    coeffs: list of coefficients
    """
    n = len(exps)
    if n == 0:
        return 0
    
    # Build the square terms as a dict
    sq_terms = {}
    for i in range(n):
        for j in range(i, n):
            deg = exps[i] + exps[j]
            if i == j:
                coeff = coeffs[i] * coeffs[i]
            else:
                coeff = 2 * coeffs[i] * coeffs[j]
            if coeff != 0:
                sq_terms[deg] = sq_terms.get(deg, 0) + coeff
    
    return sum(1 for v in sq_terms.values() if v != 0)

def _local_refine(exps, coeffs, time_limit=5.0):
    """
    Targeted local search: try small changes to exponents and coefficients
    to reduce nonzero count in the square.
    """
    start = time.time()
    best_exps = list(exps)
    best_coeffs = list(coeffs)
    best_score = _square_nonzero_count_fast(best_exps, best_coeffs)
    
    coeff_choices = [-8, -4, -2, -1, 1, 2, 4, 8, -16, 16, -32, 32, -64, 64]
    
    while time.time() - start < time_limit:
        improved = False
        # Try changing each coefficient
        for i in range(12):
            for nc in coeff_choices:
                if nc == best_coeffs[i]:
                    continue
                test_coeffs = list(best_coeffs)
                test_coeffs[i] = nc
                sc = _square_nonzero_count_fast(best_exps, test_coeffs)
                if sc < best_score:
                    best_coeffs = test_coeffs
                    best_score = sc
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try changing each exponent (small moves)
        for i in range(12):
            for delta in [-2, -1, 1, 2]:
                new_exp = best_exps[i] + delta
                if new_exp < 0 or new_exp > 40:
                    continue
                # Check collision
                if any(best_exps[j] == new_exp for j in range(12) if j != i):
                    continue
                test_exps = list(best_exps)
                test_exps[i] = new_exp
                test_exps_sorted = sorted(test_exps)
                # Remap coefficients
                idx_map = {e: idx for idx, e in enumerate(test_exps)}
                test_exps = test_exps_sorted
                test_coeffs = [best_coeffs[idx_map[e]] for e in test_exps]
                sc = _square_nonzero_count_fast(test_exps, test_coeffs)
                if sc < best_score:
                    best_exps = test_exps
                    best_coeffs = test_coeffs
                    best_score = sc
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_exps, best_coeffs, best_score

def _validate(poly):
    """Check if poly meets all constraints."""
    if len(poly) != 12:
        return False
    if any(c == 0 for _, c in poly):
        return False
    if any(c < -64 or c > 64 for _, c in poly):
        return False
    if any(e < 0 or not isinstance(e, int) for e, _ in poly):
        return False
    if any(e > 40 for e, _ in poly):
        return False
    # Check for duplicate exponents
    exps = [e for e, _ in poly]
    if len(set(exps)) != 12:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s limit
    
    best_poly = None
    best_score = float('inf')
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate diverse initial candidates
    def _generate_candidates():
        candidates = []
        # Consecutive with all 1s
        candidates.append((list(range(12)), [1]*12))
        # Consecutive with alternating signs
        candidates.append((list(range(12)), [1 if i%2==0 else -1 for i in range(12)]))
        # Arithmetic progressions
        for step in [1, 2, 3, 4, 5]:
            max_start = 40 - 11*step
            if max_start >= 0:
                start = random.randint(0, max_start)
                exps = [start + i*step for i in range(12)]
                coeffs = [random.choice([-1, 1, 2, -2]) for _ in range(12)]
                candidates.append((exps, coeffs))
        # Symmetric structures
        for _ in range(20):
            center = random.randint(6, 34)
            offsets = random.sample(range(1, 20), 6)
            exps = sorted([center - o for o in offsets] + [center + o for o in offsets])
            coeffs = [random.choice([-1, 1, 2, -2, 3, -3]) for _ in range(12)]
            candidates.append((exps, coeffs))
        # Random
        for _ in range(30):
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            candidates.append((exps, coeffs))
        # Structured: pairs with same exponent offset
        for _ in range(10):
            base = random.sample(range(0, 41), 6)
            exps = sorted(base + [b+1 for b in base])
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            candidates.append((exps, coeffs))
        return candidates
    
    # Evaluate all candidates
    candidates = _generate_candidates()
    for exps, coeffs in candidates:
        if any(e < 0 or e > 40 for e in exps):
            continue
        if len(set(exps)) != 12:
            continue
        if any(c == 0 or c < -64 or c > 64 for c in coeffs):
            continue
        sc = _square_nonzero_count_fast(exps, coeffs)
        if sc < best_score:
            best_score = sc
            best_poly = list(zip(exps, coeffs))
    
    # Genetic Algorithm
    def _ga_optimize(initial_exps, initial_coeffs, pop_size=40, generations=80):
        """Genetic algorithm to minimize nonzero terms in square."""
        start = time.time()
        time_per_gen = (deadline - start) / generations
        
        # Initialize population
        population = []
        # Elitism: include best
        population.append((list(initial_exps), list(initial_coeffs)))
        # Add diverse individuals
        for _ in range(pop_size - 1):
            if random.random() < 0.3:
                # Mutate best
                exps = list(initial_exps)
                coeffs = list(initial_coeffs)
                for _ in range(random.randint(1, 3)):
                    if random.random() < 0.5:
                        i = random.randint(0, 11)
                        new_exp = random.randint(0, 40)
                        if new_exp not in exps:
                            exps[i] = new_exp
                            exps_sorted = sorted(exps)
                            idx_map = {e: idx for idx, e in enumerate(exps)}
                            exps = exps_sorted
                            coeffs = [coeffs[idx_map[e]] for e in exps]
                    else:
                        i = random.randint(0, 11)
                        coeffs[i] = random.choice([-8, -4, -2, -1, 1, 2, 4, 8])
                population.append((exps, coeffs))
            else:
                # Random
                exps = sorted(random.sample(range(0, 41), 12))
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                population.append((exps, coeffs))
        
        # Evaluate
        for i, (exps, coeffs) in enumerate(population):
            population[i] = (exps, coeffs, _square_nonzero_count_fast(exps, coeffs))
        
        population.sort(key=lambda x: x[2])
        best_exps, best_coeffs, best_sc = population[0]
        
        elite_size = max(4, pop_size // 5)
        
        for gen in range(generations):
            if time.time() > deadline:
                break
            
            # Selection: tournament
            def _select():
                idxs = random.sample(range(len(population)), min(5, len(population)))
                best_idx = min(idxs, key=lambda i: population[i][2])
                return population[best_idx]
            
            # Create new population
            new_pop = []
            # Elitism
            for i in range(elite_size):
                new_pop.append(population[i])
            
            while len(new_pop) < pop_size:
                # Crossover + mutation
                p1 = _select()
                p2 = _select()
                
                if random.random() < 0.7:
                    # Crossover: blend exponents and coefficients
                    child_exps = list(p1[0])
                    child_coeffs = list(p1[1])
                    for i in range(12):
                        if random.random() < 0.5:
                            child_exps[i] = p2[0][i]
                        if random.random() < 0.5:
                            child_coeffs[i] = p2[1][i]
                    # Sort and remap
                    child_exps_sorted = sorted(child_exps)
                    if len(set(child_exps_sorted)) == 12:
                        idx_map = {e: idx for idx, e in enumerate(child_exps)}
                        child_exps = child_exps_sorted
                        child_coeffs = [child_coeffs[idx_map[e]] for e in child_exps]
                    else:
                        # Invalid, use parent
                        child_exps = list(p1[0])
                        child_coeffs = list(p1[1])
                else:
                    child_exps = list(p1[0])
                    child_coeffs = list(p1[1])
                
                # Mutation
                for _ in range(random.randint(1, 4)):
                    if random.random() < 0.4:
                        i = random.randint(0, 11)
                        new_exp = random.randint(0, 40)
                        if new_exp not in child_exps:
                            child_exps[i] = new_exp
                            child_exps_sorted = sorted(child_exps)
                            if len(set(child_exps_sorted)) == 12:
                                idx_map = {e: idx for idx, e in enumerate(child_exps)}
                                child_exps = child_exps_sorted
                                child_coeffs = [child_coeffs[idx_map[e]] for e in child_exps]
                    else:
                        i = random.randint(0, 11)
                        child_coeffs[i] = random.choice([-8, -4, -2, -1, 1, 2, 4, 8, -16, 16])
                
                if any(e < 0 or e > 40 for e in child_exps):
                    continue
                if len(set(child_exps)) != 12:
                    continue
                if any(c == 0 or c < -64 or c > 64 for c in child_coeffs):
                    continue
                
                sc = _square_nonzero_count_fast(child_exps, child_coeffs)
                new_pop.append((child_exps, child_coeffs, sc))
            
            population = new_pop
            population.sort(key=lambda x: x[2])
            best_exps, best_coeffs, best_sc = population[0]
            
            if best_sc <= 12:
                break
        
        return best_exps, best_coeffs, best_sc
    
    # Run GA from best initial
    if best_poly is not None:
        best_exps = [e for e, _ in best_poly]
        best_coeffs = [c for _, c in best_poly]
        
        ga_exps, ga_coeffs, ga_score = _ga_optimize(best_exps, best_coeffs, pop_size=50, generations=100)
        if ga_score < best_score:
            best_score = ga_score
            best_poly = list(zip(ga_exps, ga_coeffs))
        
        # Local refinement on GA result
        if time.time() < deadline - 5:
            lr_exps, lr_coeffs, lr_score = _local_refine(ga_exps, ga_coeffs, time_limit=min(10, deadline - time.time() - 2))
            if lr_score < best_score:
                best_score = lr_score
                best_poly = list(zip(lr_exps, lr_coeffs))
        
        # Try GA from a few other good candidates
        for exps, coeffs in candidates[:10]:
            if time.time() > deadline - 10:
                break
            sc = _square_nonzero_count_fast(exps, coeffs)
            if sc < best_score + 3:  # Only try if close
                ga_exps2, ga_coeffs2, ga_score2 = _ga_optimize(exps, coeffs, pop_size=30, generations=40)
                if ga_score2 < best_score:
                    best_score = ga_score2
                    best_poly = list(zip(ga_exps2, ga_coeffs2))
                    # Local refine
                    if time.time() < deadline - 3:
                        lr_exps2, lr_coeffs2, lr_score2 = _local_refine(ga_exps2, ga_coeffs2, time_limit=min(5, deadline - time.time() - 1))
                        if lr_score2 < best_score:
                            best_score = lr_score2
                            best_poly = list(zip(lr_exps2, lr_coeffs2))
    
    # Final local refinement
    if best_poly is not None and time.time() < deadline - 2:
        final_exps = [e for e, _ in best_poly]
        final_coeffs = [c for _, c in best_poly]
        fl_exps, fl_coeffs, fl_score = _local_refine(final_exps, final_coeffs, time_limit=min(5, deadline - time.time() - 1))
        if fl_score < best_score:
            best_score = fl_score
            best_poly = list(zip(fl_exps, fl_coeffs))
    
    # Convert to coefficient list
    if best_poly is None:
        best_poly = [(i, 1) for i in range(12)]
    
    max_deg = max(e for e, _ in best_poly)
    coeff_list = [0] * (max_deg + 1)
    for exp, coeff in best_poly:
        coeff_list[exp] = coeff
    
    return coeff_list
