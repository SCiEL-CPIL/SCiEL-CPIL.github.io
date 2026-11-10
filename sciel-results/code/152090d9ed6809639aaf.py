import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 78.0

    best_result = [1] * 12
    best_nonzero = 144

    def count_nonzero_square(coeffs, exponents):
        n = len(coeffs)
        max_deg = max(exponents)
        size = 2 * max_deg + 1
        square = np.zeros(size, dtype=np.int64)
        for i in range(n):
            for j in range(i, n):
                d = exponents[i] + exponents[j]
                val = coeffs[i] * coeffs[j]
                if i == j:
                    square[d] += val
                else:
                    square[d] += 2 * val
        return int(np.count_nonzero(square))

    def is_valid_result(coeffs, exponents):
        if len(coeffs) != 12:
            return False
        if any(c == 0 for c in coeffs):
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        if any(e < 0 for e in exponents):
            return False
        max_deg = max(exponents)
        if 2 * max_deg + 1 > 41:
            return False
        return True

    def coeffs_from_state(coeffs, exponents):
        max_deg = max(exponents)
        result = [0] * (max_deg + 1)
        for c, e in zip(coeffs, exponents):
            result[e] = c
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result

    def update_best(coeffs, exponents, nz):
        nonlocal best_nonzero, best_result
        if nz < best_nonzero:
            best_nonzero = nz
            result = coeffs_from_state(coeffs, exponents)
            if len(result) <= 41 and result[-1] != 0:
                best_result = result

    # Spectral/autocorrelation-based exponent pattern discovery
    # For a polynomial P with support S, the square P^2 has support in S+S.
    # We want S+S to have as few distinct sums as possible (dense collisions).
    # The number of distinct sums in S+S is minimized when S is an arithmetic progression
    # or a subset of one. But we also want the collision structure to allow cancellations.
    
    # Key insight: if S+S has exactly k distinct values, then P^2 has at most k terms.
    # For 12 elements, the minimum distinct sums is 23 (AP). But with clever coefficient
    # choices, many of those 23 can cancel to zero.
    
    # Strategy: enumerate exponent sets with high collision density, then solve for
    # coefficients that maximize cancellations using a targeted approach.
    
    # Generate candidate exponent sets
    exponent_sets = []
    
    # 1. Arithmetic progressions
    for step in [1, 2, 3]:
        for start in range(0, 21 - 11 * step + 1):
            exps = [start + i * step for i in range(12)]
            if max(exps) * 2 + 1 <= 41:
                exponent_sets.append(exps)
    
    # 2. Two-cluster: 6 in [0..5], 6 in [k..k+5]
    for k in range(6, 16):
        exps = list(range(6)) + list(range(k, k + 6))
        if max(exps) * 2 + 1 <= 41:
            exponent_sets.append(exps)
    
    # 3. Three-cluster
    for k1 in range(6, 10):
        for k2 in range(k1 + 6, 16):
            exps = list(range(6)) + list(range(k1, k1 + 6)) + list(range(k2, k2 + 6))
            if len(exps) == 12 and max(exps) * 2 + 1 <= 41:
                exponent_sets.append(exps)
    
    # 4. Random sets with high collision density
    rng_exp = random.Random(42)
    for _ in range(200):
        # Generate sets biased toward small values (more collisions)
        exps = sorted(rng_exp.sample(range(0, 21), 12))
        if max(exps) * 2 + 1 <= 41:
            # Count distinct sums
            sums = set()
            for i in range(12):
                for j in range(i, 12):
                    sums.add(exps[i] + exps[j])
            if len(sums) <= 30:  # High collision density
                exponent_sets.append(exps)
    
    # 5. Structured patterns from known good constructions
    exponent_sets.extend([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20],
        [0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20],
    ])
    
    # Remove duplicates
    seen = set()
    unique_exponent_sets = []
    for exps in exponent_sets:
        key = tuple(exps)
        if key not in seen:
            seen.add(key)
            unique_exponent_sets.append(exps)
    exponent_sets = unique_exponent_sets
    
    # For each exponent set, try to find coefficients that minimize nonzero square terms
    # Using a combination of:
    # 1. Systematic sign search with fixed magnitudes
    # 2. Local search with multiple perturbations
    # 3. Targeted cancellation: identify which square terms can be zeroed
    
    for exponents in exponent_sets:
        if time.time() > deadline:
            break
        
        n = 12
        max_deg = max(exponents)
        
        # Build collision map: for each degree d, which (i,j) pairs contribute
        degree_pairs = defaultdict(list)
        for i in range(n):
            for j in range(i, n):
                d = exponents[i] + exponents[j]
                degree_pairs[d].append((i, j))
        
        # Number of distinct degrees in the square
        num_degrees = len(degree_pairs)
        
        # Quick upper bound: if num_degrees is already very small, focus here
        # For each degree with multiple pairs, we can potentially cancel
        
        # Strategy A: All ±1 coefficients, search over signs
        # Use simulated annealing / local search
        rng_local = random.Random(hash(tuple(exponents)) % 100000)
        
        best_signs = None
        best_nz_signs = 999
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            signs = [1 if rng_local.random() < 0.5 else -1 for _ in range(12)]
            coeffs = signs[:]
            nz = count_nonzero_square(coeffs, exponents)
            
            # Local search: flip individual signs
            improved = True
            for _ in range(30):
                if time.time() > deadline:
                    break
                if not improved:
                    break
                improved = False
                cur_nz = nz
                for idx in range(12):
                    new_coeffs = coeffs[:]
                    new_coeffs[idx] = -new_coeffs[idx]
                    new_nz = count_nonzero_square(new_coeffs, exponents)
                    if new_nz < cur_nz:
                        cur_nz = new_nz
                        coeffs = new_coeffs
                        improved = True
                if improved:
                    nz = cur_nz
            
            if nz < best_nz_signs:
                best_nz_signs = nz
                best_signs = coeffs[:]
            
            if best_nz_signs <= 15:
                break
        
        if best_signs:
            update_best(best_signs, exponents, best_nz_signs)
        
        # Strategy B: Mixed magnitudes with local search
        mag_options = [[1]*12,
                       [1,1,1,1,1,1,2,2,2,2,2,2],
                       [1,1,2,2,3,3,4,4,5,5,6,6],
                       [1,2,1,2,1,2,1,2,1,2,1,2],
                       [1,1,1,2,2,2,3,3,3,4,4,4]]
        
        for mags in mag_options:
            if time.time() > deadline:
                break
            
            for trial in range(20):
                if time.time() > deadline:
                    break
                
                signs = [1 if rng_local.random() < 0.5 else -1 for _ in range(12)]
                coeffs = [mags[i] * signs[i] for i in range(12)]
                nz = count_nonzero_square(coeffs, exponents)
                
                # Local search: flip signs and adjust magnitudes
                improved = True
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    if not improved:
                        break
                    improved = False
                    cur_nz = nz
                    
                    # Try sign flips
                    for idx in range(12):
                        new_coeffs = coeffs[:]
                        new_coeffs[idx] = -new_coeffs[idx]
                        new_nz = count_nonzero_square(new_coeffs, exponents)
                        if new_nz < cur_nz:
                            cur_nz = new_nz
                            coeffs = new_coeffs
                            improved = True
                    
                    # Try magnitude changes
                    for idx in range(12):
                        for delta in [-1, 1]:
                            new_mag = abs(coeffs[idx]) + delta
                            if 1 <= new_mag <= 8:
                                new_coeffs = coeffs[:]
                                new_coeffs[idx] = np.sign(coeffs[idx]) * new_mag
                                new_nz = count_nonzero_square(new_coeffs, exponents)
                                if new_nz < cur_nz:
                                    cur_nz = new_nz
                                    coeffs = new_coeffs
                                    improved = True
                    
                    if improved:
                        nz = cur_nz
                
                if is_valid_result(coeffs, exponents):
                    update_best(coeffs, exponents, nz)
        
        # Strategy C: Targeted cancellation using linear algebra
        # For degrees with exactly 2 pairs, we can often cancel by choosing signs
        # For degrees with 3+ pairs, it's harder but possible with right magnitudes
        
        # Identify "easy" cancellation degrees
        easy_cancellations = 0
        for d, pairs in degree_pairs.items():
            if len(pairs) == 2:
                easy_cancellations += 1
        
        # If many easy cancellations, this is a promising structure
        if easy_cancellations >= 5:
            # More intensive search
            for trial in range(100):
                if time.time() > deadline:
                    break
                
                coeffs = []
                for i in range(12):
                    mag = rng_local.choice([1, 1, 2, 2, 3])
                    sign = 1 if rng_local.random() < 0.5 else -1
                    coeffs.append(sign * mag)
                
                if not is_valid_result(coeffs, exponents):
                    continue
                
                nz = count_nonzero_square(coeffs, exponents)
                
                # Aggressive local search
                for _ in range(40):
                    if time.time() > deadline:
                        break
                    improved = False
                    cur_nz = nz
                    for idx in range(12):
                        # Sign flip
                        new_coeffs = coeffs[:]
                        new_coeffs[idx] = -new_coeffs[idx]
                        new_nz = count_nonzero_square(new_coeffs, exponents)
                        if new_nz < cur_nz:
                            cur_nz = new_nz
                            coeffs = new_coeffs
                            improved = True
                        # Magnitude change
                        for delta in [-1, 1]:
                            new_mag = abs(coeffs[idx]) + delta
                            if 1 <= new_mag <= 8:
                                new_coeffs = coeffs[:]
                                new_coeffs[idx] = np.sign(coeffs[idx]) * new_mag
                                new_nz = count_nonzero_square(new_coeffs, exponents)
                                if new_nz < cur_nz:
                                    cur_nz = new_nz
                                    coeffs = new_coeffs
                                    improved = True
                    if not improved:
                        break
                    nz = cur_nz
                
                if is_valid_result(coeffs, exponents):
                    update_best(coeffs, exponents, nz)
    
    # Final refinement: take best result and try to improve it
    if best_nonzero > 15 and time.time() < deadline:
        nz_indices = [i for i, c in enumerate(best_result) if c != 0]
        if len(nz_indices) == 12:
            best_exponents = nz_indices
            best_coeffs = [best_result[i] for i in nz_indices]
            
            rng_final = random.Random(9999)
            for _ in range(500):
                if time.time() > deadline:
                    break
                
                coeffs = best_coeffs[:]
                # Random perturbation
                num_changes = rng_final.randint(1, 3)
                for _ in range(num_changes):
                    idx = rng_final.randint(0, 11)
                    choice = rng_final.random()
                    if choice < 0.4:
                        # Sign flip
                        coeffs[idx] = -coeffs[idx]
                    elif choice < 0.7:
                        # Magnitude change
                        delta = rng_final.choice([-2, -1, 1, 2])
                        new_mag = abs(coeffs[idx]) + delta
                        if 1 <= new_mag <= 16:
                            coeffs[idx] = np.sign(coeffs[idx]) * new_mag
                    else:
                        # New random value
                        mag = rng_final.randint(1, 8)
                        sign = 1 if rng_final.random() < 0.5 else -1
                        coeffs[idx] = sign * mag
                
                if any(c == 0 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, best_exponents):
                    continue
                
                nz = count_nonzero_square(coeffs, best_exponents)
                if nz < best_nonzero:
                    update_best(coeffs, best_exponents, nz)
                    best_coeffs = coeffs[:]
                
                # Also try local search from this point
                if nz <= best_nonzero + 2:
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        improved = False
                        cur_nz = nz
                        for idx in range(12):
                            new_coeffs = coeffs[:]
                            new_coeffs[idx] = -new_coeffs[idx]
                            new_nz = count_nonzero_square(new_coeffs, best_exponents)
                            if new_nz < cur_nz:
                                cur_nz = new_nz
                                coeffs = new_coeffs
                                improved = True
                        if not improved:
                            break
                        nz = cur_nz
                        if is_valid_result(coeffs, best_exponents):
                            update_best(coeffs, best_exponents, nz)
                            best_coeffs = coeffs[:]
    
    # Ensure result is valid
    if len(best_result) > 41 or best_result[-1] == 0:
        best_result = [1] * 12
    nz_count = sum(1 for c in best_result if c != 0)
    if nz_count != 12:
        best_result = [1] * 12
    if any(c < -64 or c > 64 for c in best_result):
        best_result = [1] * 12
    
    return best_result
