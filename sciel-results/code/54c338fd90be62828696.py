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

    def generate_candidate_exponents():
        candidates = []
        # 1. Arithmetic Progressions
        for k in range(1, 5):
            max_start = 40 - 11 * k
            for start in range(0, max_start + 1):
                exps = [start + i * k for i in range(12)]
                candidates.append(exps)
        
        # 2. Blocks of consecutive integers (e.g., 0,1,2, 4,5,6, 8,9,10, 12,13,14)
        # This creates structure within blocks and gaps between blocks.
        for gap in range(1, 5):
            for block_size in [2, 3, 4]:
                if 12 % block_size != 0:
                    continue
                num_blocks = 12 // block_size
                # Total span = num_blocks * block_size + (num_blocks - 1) * gap
                # We need to fit in 40
                # Start at 0
                exps = []
                for b in range(num_blocks):
                    base = b * (block_size + gap)
                    for i in range(block_size):
                        e = base + i
                        if e <= 40:
                            exps.append(e)
                if len(exps) == 12:
                    candidates.append(exps)
        
        # 3. Symmetric sets around a center
        for center in range(5, 36):
            for spread in range(1, 6):
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
                        candidates.append(exps)

        # 4. Random perturbations of APs
        random.seed(42)
        for _ in range(200):
            k = random.randint(1, 4)
            max_start = 40 - 11 * k
            start = random.randint(0, max_start)
            exps = [start + i * k for i in range(12)]
            # Perturb 1-2 exponents
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, 11)
                shift = random.randint(-2, 2)
                new_e = exps[idx] + shift
                if 0 <= new_e <= 40 and new_e not in exps:
                    exps[idx] = new_e
            exps.sort()
            if len(set(exps)) == 12:
                candidates.append(exps)
                
        # Deduplicate
        unique_candidates = []
        seen = set()
        for c in candidates:
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                unique_candidates.append(c)
        
        return unique_candidates

    def optimize_coefficients(exponents, deadline):
        n = 12
        coeff_values = [-3, -2, -1, 1, 2, 3]
        best_coeffs = None
        best_count = float('inf')
        
        # Multi-start local search
        random.seed(hash(tuple(exponents)) % 100000)
        
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Initialize with random small integers
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

    # Phase 1: Structured search
    candidates = generate_candidate_exponents()
    
    # Sort candidates by potential (e.g., number of pair sum collisions)
    def score_exponents(exps):
        sums = defaultdict(int)
        for i in range(12):
            for j in range(i+1, 12):
                sums[exps[i] + exps[j]] += 1
        collisions = sum(1 for s in sums.values() if s > 1)
        return -collisions # Lower is better for sorting (more collisions)

    candidates.sort(key=score_exponents)
    
    # Try top candidates
    for exps in candidates[:50]:
        if time.time() > deadline:
            break
        coeffs, count = optimize_coefficients(exps, deadline)
        if coeffs is not None and count < best_count:
            update_best(exps, coeffs, count)
            if best_count == 12:
                return _build_result(best_exponents, best_coeffs)
    
    # Phase 2: Global random search with local polish
    random.seed(99999)
    current_exponents = best_exponents[:]
    current_coeffs = best_coeffs[:]
    current_count = best_count
    coeff_values = [-3, -2, -1, 1, 2, 3]
    
    while time.time() < deadline:
        if random.random() < 0.2:
            # Restart with a random candidate
            exps = random.choice(candidates) if candidates else list(range(12))
            coeffs = [random.choice(coeff_values) for _ in range(12)]
            current_exponents = exps
            current_coeffs = coeffs
            current_count = count_nonzero_square(current_exponents, current_coeffs)
            update_best(current_exponents, current_coeffs, current_count)
        else:
            # Local move
            if random.random() < 0.7:
                # Change coefficient
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
            else:
                # Move exponent
                idx = random.randint(0, 11)
                old_exp = current_exponents[idx]
                shift = random.choice([-2, -1, 1, 2])
                new_exp = old_exp + shift
                if 0 <= new_exp <= 40 and new_exp not in current_exponents:
                    current_exponents[idx] = new_exp
                    current_exponents.sort()
                    # Re-sort coeffs to match exponents
                    # This is tricky. We need to map old coeffs to new positions.
                    # Since we just moved one exponent, we can find its new position.
                    # But the list is sorted.
                    # Let's just revert if it gets complex, or implement a swap.
                    # For now, revert to keep it simple and fast.
                    current_exponents[idx] = old_exp
    
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
