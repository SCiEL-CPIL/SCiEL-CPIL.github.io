import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_count = float('inf')
    best_coeffs = [1] * 12
    best_exponents = list(range(12))
    
    def count_nonzero_square(exponents, coeffs):
        n = len(exponents)
        if n == 0:
            return 0
        max_deg = max(exponents)
        square = [0] * (2 * max_deg + 1)
        for i in range(n):
            ci = coeffs[i]
            ei = exponents[i]
            if ci == 0:
                continue
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                cj = coeffs[j]
                ej = exponents[j]
                if cj == 0:
                    continue
                prod = 2 * ci * cj
                square[ei + ej] += prod
        return sum(1 for x in square if x != 0)

    def update_best(exponents, coeffs, count):
        nonlocal best_count, best_coeffs, best_exponents
        if count < best_count:
            best_count = count
            best_exponents = exponents[:]
            best_coeffs = coeffs[:]
            if best_count == 12:
                return True
        return False

    def find_cancellation_structures():
        """
        Find exponent sets where many pairs (i,j) share the same sum,
        enabling coefficient cancellations.
        
        For a given set of 12 exponents, group pairs by their sum.
        If a sum s has multiple pairs (i,j) with i+j=s, we can potentially
        cancel them by choosing coefficients such that sum of c_i*c_j = 0.
        
        We score exponent sets by how many "redundant" pair-sums they have
        (sums that appear more than once among distinct pairs).
        """
        results = []
        
        # Try structured exponent patterns
        # Pattern 1: Two interleaved APs
        # e.g., {0,2,4,...,22} and {1,3,5,...,23} -> sums overlap significantly
        for offset1 in range(0, 4):
            for offset2 in range(0, 4):
                for step in [1, 2, 3]:
                    exps = []
                    # First 6 terms: offset1 + i*step
                    for i in range(6):
                        e = offset1 + i * step
                        if e <= 40:
                            exps.append(e)
                    # Next 6 terms: offset2 + i*step
                    for i in range(6):
                        e = offset2 + i * step
                        if e <= 40:
                            exps.append(e)
                    if len(exps) == 12:
                        exps.sort()
                        if len(set(exps)) == 12:
                            # Score by number of pair-sum collisions
                            sums = defaultdict(int)
                            for i in range(12):
                                for j in range(i+1, 12):
                                    sums[exps[i] + exps[j]] += 1
                            collisions = sum(1 for s in sums.values() if s > 1)
                            total_pairs = 66
                            results.append((collisions, exps[:]))
        
        # Pattern 2: Symmetric around center
        for center in range(10, 31):
            for spread in range(1, 10):
                exps = []
                for i in range(6):
                    e1 = center - i * spread
                    e2 = center + i * spread
                    if 0 <= e1 <= 40:
                        exps.append(e1)
                    if 0 <= e2 <= 40 and e2 != e1:
                        exps.append(e2)
                if len(exps) >= 12:
                    exps = sorted(set(exps))[:12]
                    if len(exps) == 12:
                        sums = defaultdict(int)
                        for i in range(12):
                            for j in range(i+1, 12):
                                sums[exps[i] + exps[j]] += 1
                        collisions = sum(1 for s in sums.values() if s > 1)
                        results.append((collisions, exps[:]))
        
        # Pattern 3: Random with collision bias
        random.seed(42)
        for _ in range(500):
            exps = random.sample(range(41), 12)
            exps.sort()
            sums = defaultdict(int)
            for i in range(12):
                for j in range(i+1, 12):
                    sums[exps[i] + exps[j]] += 1
            collisions = sum(1 for s in sums.values() if s > 1)
            results.append((collisions, exps[:]))
        
        # Sort by collisions descending
        results.sort(key=lambda x: -x[0])
        return results

    def solve_coefficients_for_exponents(exponents, deadline):
        """
        Given exponents, find coefficients that maximize cancellation.
        Uses constraint-based local search.
        """
        n = 12
        coeff_values = [-2, -1, 1, 2]
        
        # Group pairs by sum
        pair_sums = defaultdict(list)
        for i in range(n):
            for j in range(i+1, n):
                pair_sums[exponents[i] + exponents[j]].append((i, j))
        
        best_coeffs = None
        best_count = float('inf')
        
        # Multi-start local search
        random.seed(hash(tuple(exponents)) % 100000)
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            coeffs = [random.choice(coeff_values) for _ in range(n)]
            count = count_nonzero_square(exponents, coeffs)
            
            # Local search: try changing individual coefficients
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(n):
                    old_val = coeffs[i]
                    for new_val in coeff_values:
                        if new_val == old_val:
                            continue
                        coeffs[i] = new_val
                        new_count = count_nonzero_square(exponents, coeffs)
                        if new_count < count:
                            count = new_count
                            improved = True
                        else:
                            coeffs[i] = old_val
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            
            if count < best_count:
                best_count = count
                best_coeffs = coeffs[:]
                if best_count == 12:
                    break
        
        return best_coeffs, best_count

    # Phase 1: Find good exponent structures
    if time.time() > deadline:
        return _build_result(best_exponents, best_coeffs)
    
    structures = find_cancellation_structures()
    
    # Try top structures with coefficient solving
    for idx, (collisions, exps) in enumerate(structures[:20]):
        if time.time() > deadline:
            break
        
        coeffs, count = solve_coefficients_for_exponents(exps, deadline)
        if coeffs is not None and count < best_count:
            update_best(exps, coeffs, count)
            if best_count == 12:
                return _build_result(best_exponents, best_coeffs)
    
    # Phase 2: Fallback to AP search with sign patterns
    for k in range(1, 4):
        max_start = 40 - 11 * k
        for start in range(0, max_start + 1):
            if time.time() > deadline:
                break
            exponents = [start + i * k for i in range(12)]
            for bits in range(1 << 12):
                if time.time() > deadline:
                    break
                coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
                count = count_nonzero_square(exponents, coeffs)
                if update_best(exponents, coeffs, count):
                    if best_count == 12:
                        return _build_result(best_exponents, best_coeffs)
    
    # Phase 3: Final random polish
    random.seed(99999)
    current_exponents = best_exponents[:]
    current_coeffs = best_coeffs[:]
    current_count = best_count
    coeff_values = [-2, -1, 1, 2]
    
    while time.time() < deadline:
        if random.random() < 0.3:
            exps = random.sample(range(41), 12)
            exps.sort()
            coeffs = [random.choice(coeff_values) for _ in range(12)]
            current_exponents = exps
            current_coeffs = coeffs
            current_count = count_nonzero_square(current_exponents, current_coeffs)
            update_best(current_exponents, current_coeffs, current_count)
        else:
            idx = random.randint(0, 11)
            old_val = current_coeffs[idx]
            new_val = random.choice(coeff_values)
            if new_val != old_val:
                current_coeffs[idx] = new_val
                new_count = count_nonzero_square(current_exponents, current_coeffs)
                if new_count < current_count:
                    current_count = new_count
                    update_best(current_exponents, current_coeffs, new_count)
                elif random.random() < 0.1:
                    current_count = new_count
                else:
                    current_coeffs[idx] = old_val
    
    return _build_result(best_exponents, best_coeffs)

def _build_result(exponents, coeffs):
    max_deg = max(exponents) if exponents else 0
    result = [0] * (max_deg + 1)
    for e, c in zip(exponents, coeffs):
        result[e] = c
    if len(result) > 41:
        result = result[:41]
    if result and result[-1] == 0:
        for i in range(len(result) - 1, -1, -1):
            if result[i] != 0:
                result = result[:i+1]
                break
    if not result:
        result = [1]
    return result
