import time
import numpy as np
import random
from itertools import product

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_score_fast(exps, coeffs):
        """
        Fast computation of nonzero terms in P(x)^2.
        exps: list of 12 exponents (sorted)
        coeffs: list of 12 coefficients
        Returns count of nonzero terms, or inf if invalid.
        """
        if not exps or not coeffs:
            return float('inf')

        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')

        # Build dense array
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c

        q = np.convolve(p, p)
        count = int(np.count_nonzero(q))
        return count

    def _update_best(exps, coeffs, score):
        """Update the global best candidate if improved."""
        nonlocal best_score, best_candidate
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def factor_product_score(factors):
        """
        Given 3 factors, each a list of (exp, coeff) pairs (4 each),
        compute the number of nonzero terms in P(x)^2 where P = F1*F2*F3.
        """
        # Build F1, F2, F3 as dense arrays
        max_exp = max(max(e for e, c in f) for f in factors)
        # P = F1*F2*F3, max degree = 3*max_exp
        # But we need max degree of P <= 40, so 3*max_exp <= 40 => max_exp <= 13
        
        if max_exp > 13:
            return float('inf')
        
        # Build each factor
        arrays = []
        for f in factors:
            arr = np.zeros(max_exp + 1, dtype=np.int64)
            for e, c in f:
                if 0 <= e <= max_exp and abs(c) <= 64 and c != 0:
                    arr[e] = c
                else:
                    return float('inf')
            arrays.append(arr)
        
        # P = F1 * F2 * F3
        p = np.convolve(np.convolve(arrays[0], arrays[1]), arrays[2])
        
        # Check constraints on P
        if len(p) > 41:
            return float('inf')
        if p[-1] == 0:
            # Trim trailing zeros
            while len(p) > 1 and p[-1] == 0:
                p = p[:-1]
            if p[-1] == 0:
                return float('inf')
        
        # Count nonzero terms in P
        nonzero_in_p = np.count_nonzero(p)
        if nonzero_in_p != 12:
            return float('inf')
        
        # Check coefficients
        if any(abs(c) > 64 for c in p):
            return float('inf')
        
        # Compute P^2
        q = np.convolve(p, p)
        return int(np.count_nonzero(q))

    def optimize_factor_structure():
        """
        Search over 3-factor decompositions.
        Each factor has 4 terms with exponents in [0, 13] (since 3*13=39 <= 40).
        We try to find factor structures where P = F1*F2*F3 has exactly 12
        nonzero terms and P^2 has minimal nonzero terms.
        """
        nonlocal best_score, best_candidate
        
        coeff_choices = [-1, 1, -2, 2, -3, 3, -4, 4, -8, 8]
        
        # Generate candidate factor structures
        base_structures = []
        
        # Structure 1: Consecutive small exponents
        for s1 in range(0, 3):
            for s2 in range(0, 3):
                for s3 in range(0, 3):
                    f1 = [(s1 + i, 1) for i in range(4)]
                    f2 = [(s2 + i, 1) for i in range(4)]
                    f3 = [(s3 + i, 1) for i in range(4)]
                    base_structures.append((f1, f2, f3))
        
        # Structure 2: Gapped exponents
        for gap in [2, 3, 4]:
            for offset in range(0, 5):
                f1 = [(offset + i*gap, 1) for i in range(4)]
                f2 = [(offset + i*gap, 1) for i in range(4)]
                f3 = [(offset + i*gap, 1) for i in range(4)]
                if max(e for e, c in f3) <= 13:
                    base_structures.append((f1, f2, f3))
        
        # Structure 3: Mixed
        f1 = [(0, 1), (1, 1), (2, 1), (4, 1)]
        f2 = [(0, 1), (1, 1), (2, 1), (3, 1)]
        f3 = [(0, 1), (1, 1), (2, 1), (3, 1)]
        base_structures.append((f1, f2, f3))
        
        # For each structure, try different coefficient patterns
        for f1, f2, f3 in base_structures:
            if time.time() > deadline - 15:
                break
            
            # Try all sign patterns for the 12 coefficients
            # That's 2^12 = 4096 patterns - feasible
            for signs in product([-1, 1], repeat=12):
                if time.time() > deadline - 10:
                    break
                
                # Apply signs to factors
                sf1 = [(e, c * signs[i]) for i, (e, c) in enumerate(f1)]
                sf2 = [(e, c * signs[i+4]) for i, (e, c) in enumerate(f2)]
                sf3 = [(e, c * signs[i+8]) for i, (e, c) in enumerate(f3)]
                
                score = factor_product_score([sf1, sf2, sf3])
                if score < float('inf') and score < best_score:
                    # Reconstruct P
                    max_exp = max(max(e for e, c in f) for f in [sf1, sf2, sf3])
                    arrs = []
                    for f in [sf1, sf2, sf3]:
                        arr = np.zeros(max_exp + 1, dtype=np.int64)
                        for e, c in f:
                            arr[e] = c
                        arrs.append(arr)
                    p = np.convolve(np.convolve(arrs[0], arrs[1]), arrs[2])
                    # Trim
                    while len(p) > 1 and p[-1] == 0:
                        p = p[:-1]
                    if len(p) <= 41 and p[-1] != 0:
                        dense = list(p)
                        best_score = score
                        best_candidate = dense
        
        # Now do local search on the best factor structure found
        # Or: do a broader search with random factor structures
        if time.time() < deadline - 10:
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Random factor structure
                f1_exps = sorted(random.sample(range(0, 8), 4))
                f2_exps = sorted(random.sample(range(0, 8), 4))
                f3_exps = sorted(random.sample(range(0, 8), 4))
                
                f1 = [(e, random.choice([-1, 1])) for e in f1_exps]
                f2 = [(e, random.choice([-1, 1])) for e in f2_exps]
                f3 = [(e, random.choice([-1, 1])) for e in f3_exps]
                
                score = factor_product_score([f1, f2, f3])
                if score < float('inf') and score < best_score:
                    max_exp = max(max(e for e, c in f) for f in [f1, f2, f3])
                    arrs = []
                    for f in [f1, f2, f3]:
                        arr = np.zeros(max_exp + 1, dtype=np.int64)
                        for e, c in f:
                            arr[e] = c
                        arrs.append(arr)
                    p = np.convolve(np.convolve(arrs[0], arrs[1]), arrs[2])
                    while len(p) > 1 and p[-1] == 0:
                        p = p[:-1]
                    if len(p) <= 41 and p[-1] != 0:
                        dense = list(p)
                        best_score = score
                        best_candidate = dense

    def local_search_from_candidate():
        """
        Given the current best candidate, do local search by modifying
        the underlying factor structure or directly modifying P.
        """
        nonlocal best_score, best_candidate
        
        if time.time() > deadline - 5:
            return
        
        # Extract exponents and coefficients from best_candidate
        exps = [i for i, c in enumerate(best_candidate) if c != 0]
        coeffs = [c for c in best_candidate if c != 0]
        
        if len(exps) != 12:
            return
        
        # Try flipping signs of individual coefficients
        T = 5.0
        T_min = 0.5
        cooling = 0.99
        
        current_score = compute_score_fast(exps, coeffs)
        
        for it in range(3000):
            if time.time() > deadline - 3:
                break
            
            i = random.randint(0, 11)
            new_coeffs = coeffs[:]
            new_coeffs[i] = -new_coeffs[i]
            
            new_score = compute_score_fast(exps, new_coeffs)
            
            if new_score < current_score:
                coeffs = new_coeffs
                current_score = new_score
                _update_best(exps, coeffs, current_score)
            else:
                delta = new_score - current_score
                if T > T_min:
                    if random.random() < np.exp(-delta / T):
                        coeffs = new_coeffs
                        current_score = new_score
            
            T = max(T * cooling, T_min)

    def genetic_algorithm(pop_size=20, generations=50):
        """
        Genetic algorithm with crossover and mutation to explore the space.
        """
        nonlocal best_score, best_candidate
        
        # Initialize population
        population = []
        
        # Seed with diverse candidates
        def generate_random_individual():
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1, -2, 2, -3, 3]) for _ in range(12)]
            return exps, coeffs
        
        # Add structured candidates
        structured_candidates = []
        for start in range(0, 5):
            exps = [start + i for i in range(12)]
            if max(exps) <= 40:
                for pattern in [
                    [1]*12,
                    [1 if i%2==0 else -1 for i in range(12)],
                    [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
                    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
                ]:
                    structured_candidates.append((exps[:], pattern[:]))
        
        # Add random candidates
        for _ in range(pop_size - len(structured_candidates)):
            structured_candidates.append(generate_random_individual())
        
        population = structured_candidates[:pop_size]
        
        # Evaluate initial population
        for exps, coeffs in population:
            score = compute_score_fast(exps, coeffs)
            _update_best(exps, coeffs, score)
        
        for gen in range(generations):
            if time.time() > deadline - 5:
                break
            
            # Evaluate population
            scored_population = []
            for exps, coeffs in population:
                score = compute_score_fast(exps, coeffs)
                scored_population.append((score, exps, coeffs))
            
            scored_population.sort(key=lambda x: x[0])
            
            # Elitism: keep top half
            elite_count = max(1, pop_size // 2)
            new_population = []
            for _, exps, coeffs in scored_population[:elite_count]:
                new_population.append((exps[:], coeffs[:]))
            
            # Generate offspring via crossover and mutation
            while len(new_population) < pop_size:
                if time.time() > deadline - 5:
                    break
                
                # Tournament selection
                def tournament_select(pop, k=3):
                    candidates = random.sample(pop, min(k, len(pop)))
                    return min(candidates, key=lambda x: x[0])
                
                parent1 = tournament_select(scored_population)
                parent2 = tournament_select(scored_population)
                
                # Crossover: uniform crossover on exponents and coefficients
                child_exps = []
                child_coeffs = []
                for i in range(12):
                    if random.random() < 0.5:
                        child_exps.append(parent1[1][i])
                        child_coeffs.append(parent1[2][i])
                    else:
                        child_exps.append(parent2[1][i])
                        child_coeffs.append(parent2[2][i])
                
                # Mutation
                if random.random() < 0.3:
                    # Change a coefficient
                    idx = random.randint(0, 11)
                    child_coeffs[idx] = random.choice([-1, 1, -2, 2, -3, 3])
                
                if random.random() < 0.2:
                    # Shift an exponent
                    idx = random.randint(0, 11)
                    delta = random.choice([-1, 1])
                    new_e = child_exps[idx] + delta
                    if 0 <= new_e <= 40 and new_e not in child_exps:
                        child_exps[idx] = new_e
                        child_exps.sort()
                        # Re-sort coefficients to match exponents
                        idxs = np.argsort(child_exps)
                        child_exps = [child_exps[i] for i in idxs]
                        child_coeffs = [child_coeffs[i] for i in idxs]
                
                new_population.append((child_exps, child_coeffs))
            
            population = new_population
        
        # Final evaluation
        for exps, coeffs in population:
            score = compute_score_fast(exps, coeffs)
            _update_best(exps, coeffs, score)

    # --- Main search loop ---

    random.seed(42)

    # Phase 1: Factor structure search
    optimize_factor_structure()

    # Phase 2: Genetic algorithm
    if time.time() < deadline - 10:
        genetic_algorithm(pop_size=20, generations=50)

    # Phase 3: Local search on the result
    local_search_from_candidate()

    # Phase 4: If we haven't found something great, try direct random search
    if best_score > 15 and time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            score = compute_score_fast(exps, coeffs)
            if score < best_score:
                max_deg = max(exps)
                dense = [0] * (max_deg + 1)
                for e, c in zip(exps, coeffs):
                    dense[e] = c
                if len(dense) <= 41 and dense[-1] != 0:
                    best_score = score
                    best_candidate = dense

    return best_candidate

# EVOLVE-BLOCK-END
