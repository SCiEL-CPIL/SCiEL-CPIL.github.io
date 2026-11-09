"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

def _count_nonzero(coeffs):
    """Count nonzero entries in a coefficient list."""
    return sum(1 for c in coeffs if c != 0)

def _get_sparsity(pairs):
    """Compute the number of nonzero terms in the square of the polynomial defined by pairs.
    pairs: list of (exponent, coefficient) tuples."""
    if not pairs:
        return 0
    max_deg = max(e for e, c in pairs if c != 0)
    arr = np.zeros(2 * max_deg + 1, dtype=int)
    for e, c in pairs:
        if c != 0:
            arr[e] += c
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Candidate exponent sets
    # 1. Arithmetic Progression (0..11)
    exps_ap = list(range(12))
    # 2. Arithmetic Progression with step 2 (0, 2, ..., 22)
    exps_ap2 = [2*i for i in range(12)]
    # 3. Clumped blocks
    exps_clump = [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32]
    # 4. Symmetric around center
    exps_sym = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19] # Just a guess
    
    candidate_exps = [exps_ap, exps_ap2, exps_clump, exps_sym]
    
    best_sparsity = float('inf')
    best_result_pairs = None
    
    # Step 1: Exhaustive search over binary coefficients for each exponent set
    # 2^12 = 4096 iterations, very fast.
    for exps in candidate_exps:
        if time.time() - start_time > time_limit:
            break
        for mask in range(1 << 12):
            coeffs = []
            for i in range(12):
                if (mask >> i) & 1:
                    coeffs.append(1)
                else:
                    coeffs.append(-1)
            pairs = list(zip(exps, coeffs))
            sparsity = _get_sparsity(pairs)
            if sparsity < best_sparsity:
                best_sparsity = sparsity
                best_result_pairs = pairs[:]
                
    # Step 2: Local search on coefficients for the best exponent set found so far
    # We will try to improve the best result found in Step 1 by allowing coefficients in {-2, -1, 1, 2}
    # and also trying to change exponents slightly if needed, but let's stick to the exponent set of the best binary solution.
    
    if best_result_pairs:
        current_pairs = best_result_pairs[:]
        current_sparsity = best_sparsity
        
        # Local search loop
        # We will perform a simple hill climbing on the coefficients.
        # We allow changes to coefficients in {-2, -1, 1, 2}.
        # We also allow swapping exponents? No, let's fix exponents from the best binary run first.
        # Actually, let's allow exponent changes too, but only small ones.
        
        for _ in range(2000):
            if time.time() - start_time > time_limit:
                break
            
            # Randomly pick a move type
            move_type = random.choice(['coeff', 'exp', 'both'])
            
            if move_type == 'coeff':
                i = random.randint(0, 11)
                e, c = current_pairs[i]
                new_c = random.choice([-2, -1, 1, 2])
                if new_c == c: continue
                new_pairs = current_pairs[:]
                new_pairs[i] = (e, new_c)
                
            elif move_type == 'exp':
                i = random.randint(0, 11)
                e, c = current_pairs[i]
                # Try to move exponent to a nearby unused slot
                used = set(p[0] for p in current_pairs)
                # Generate candidates: e-2, e-1, e+1, e+2, or random
                candidates = [e-2, e-1, e+1, e+2]
                valid_candidates = [x for x in candidates if x >= 0 and x < 41 and x not in used]
                if valid_candidates:
                    new_e = random.choice(valid_candidates)
                else:
                    # If no valid nearby, pick random unused
                    all_unused = [x for x in range(41) if x not in used]
                    if not all_unused: continue
                    new_e = random.choice(all_unused)
                new_pairs = current_pairs[:]
                new_pairs[i] = (new_e, c)
                new_pairs.sort(key=lambda x: x[0])
                
            else: # both
                i = random.randint(0, 11)
                e, c = current_pairs[i]
                new_c = random.choice([-2, -1, 1, 2])
                used = set(p[0] for p in current_pairs)
                candidates = [e-1, e+1]
                valid_candidates = [x for x in candidates if x >= 0 and x < 41 and x not in used]
                if valid_candidates:
                    new_e = random.choice(valid_candidates)
                else:
                    new_e = e
                new_pairs = current_pairs[:]
                new_pairs[i] = (new_e, new_c)
                new_pairs.sort(key=lambda x: x[0])
                
            next_sparsity = _get_sparsity(new_pairs)
            
            if next_sparsity < current_sparsity:
                current_pairs = new_pairs
                current_sparsity = next_sparsity
                if current_sparsity < best_sparsity:
                    best_sparsity = current_sparsity
                    best_result_pairs = current_pairs[:]
            elif next_sparsity == current_sparsity and random.random() < 0.1:
                # Accept equal with small probability to escape plateaus
                current_pairs = new_pairs
                current_sparsity = next_sparsity
                if current_sparsity < best_sparsity:
                    best_sparsity = current_sparsity
                    best_result_pairs = current_pairs[:]

    if best_result_pairs is None:
        # Fallback
        best_result_pairs = list(zip(range(12), [1]*12))
        best_sparsity = _get_sparsity(best_result_pairs)

    # Convert best_result_pairs to coefficient list
    max_deg = max(e for e, c in best_result_pairs)
    result = [0] * (max_deg + 1)
    for e, c in best_result_pairs:
        result[e] = c
        
    # Ensure length <= 41
    if len(result) > 41:
        result = result[:41]
        
    # Ensure nonzero final coefficient
    if result[-1] == 0:
        last_nonzero = -1
        for i in range(len(result) - 1, -1, -1):
            if result[i] != 0:
                last_nonzero = i
                break
        if last_nonzero != -1:
            result = result[:last_nonzero + 1]
        if result[-1] == 0:
            result[-1] = 1
            
    # Ensure exactly 12 nonzero terms
    nonzero_count = sum(1 for c in result if c != 0)
    if nonzero_count != 12:
        # This should not happen if we manage pairs correctly, but safety check
        # Reconstruct from pairs to be sure
        max_deg = max(e for e, c in best_result_pairs)
        result = [0] * (max_deg + 1)
        for e, c in best_result_pairs:
            result[e] = c
        if len(result) > 41:
             result = result[:41]
        if result[-1] == 0:
             # Find last nonzero
             last_nonzero = -1
             for i in range(len(result) - 1, -1, -1):
                 if result[i] != 0:
                     last_nonzero = i
                     break
             if last_nonzero != -1:
                 result = result[:last_nonzero + 1]
             if result[-1] == 0:
                 result[-1] = 1
                 
    return result

# EVOLVE-BLOCK-END
