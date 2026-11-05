import time
import random
import numpy as np

def _count_nonzero_square_fast(coeffs):
    """Count nonzero terms in the exact square using NumPy convolution."""
    if len(coeffs) == 0:
        return 0
    c = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(c, c)
    return int(np.count_nonzero(sq))

def _validate(coeffs):
    """Check all constraints on the coefficient list."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    for c in coeffs:
        if not isinstance(c, int) or c < -64 or c > 64:
            return False
    nonzero = sum(1 for c in coeffs if c != 0)
    if nonzero != 12:
        return False
    return True

def _sparse_to_dense(sparse, max_deg=40):
    """Convert sparse list of (exponent, coeff) to dense coefficient list."""
    coeffs = [0] * (max_deg + 1)
    for exp, c in sparse:
        if 0 <= exp <= max_deg:
            coeffs[exp] = c
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs

def _dense_to_sparse(coeffs):
    """Convert dense coefficient list to sparse representation."""
    return [(i, c) for i, c in enumerate(coeffs) if c != 0]

def _local_search(sparse_init, deadline, coeff_choices=None, seed=42):
    """
    Perform Simulated Annealing on the coefficients and exponents.
    """
    if coeff_choices is None:
        # Small integer coefficients are more likely to cancel exactly
        coeff_choices = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    
    current_sparse = sorted(sparse_init) # Keep sorted by exponent
    current_dense = _sparse_to_dense(current_sparse)
    current_sq = _count_nonzero_square_fast(current_dense)
    
    best_sparse = list(current_sparse)
    best_sq = current_sq
    
    random.seed(seed)
    
    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.1
    # Estimate iterations: we want to finish before deadline - 1s
    # Assume ~1000 iterations/sec
    max_iterations = int((deadline - time.time() - 1.0) * 1000)
    if max_iterations < 100:
        return best_sparse, best_sq
        
    stall_count = 0
    max_stall = 2000
    
    for it in range(max_iterations):
        if time.time() >= deadline - 1.0:
            break
        
        progress = it / max(1, max_iterations - 1)
        T = T_start * (T_end / T_start) ** progress
        
        trial_sparse = list(current_sparse)
        valid = True
        
        # Choose a move type
        # Focus on coefficients, but allow some exponent shifts
        move_type = random.choices(
            ['change_coeff', 'flip_sign', 'shift_exp', 'swap_coeff', 'shift_pair'],
            weights=[40, 20, 15, 15, 10]
        )[0]
        
        if move_type == 'change_coeff':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            new_c = random.choice(coeff_choices)
            if new_c != c:
                trial_sparse[idx] = (e, new_c)
            else:
                valid = False
                
        elif move_type == 'flip_sign':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            trial_sparse[idx] = (e, -c)
            
        elif move_type == 'shift_exp':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            shift = random.choice([-2, -1, 1, 2])
            new_e = e + shift
            if 0 <= new_e <= 40:
                existing_exps = set(x[0] for x in trial_sparse if x[0] != e)
                if new_e not in existing_exps:
                    trial_sparse[idx] = (new_e, c)
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'swap_coeff':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                trial_sparse[i] = (ei, cj)
                trial_sparse[j] = (ej, ci)
            else:
                valid = False
                
        elif move_type == 'shift_pair':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                shift = random.choice([-2, -1, 1, 2])
                new_ei = ei + shift
                new_ej = ej + shift
                if 0 <= new_ei <= 40 and 0 <= new_ej <= 40:
                    other_exps = set(x[0] for k, x in enumerate(trial_sparse) if k not in (i, j))
                    if new_ei not in other_exps and new_ej not in other_exps and new_ei != new_ej:
                        trial_sparse[i] = (new_ei, ci)
                        trial_sparse[j] = (new_ej, cj)
                    else:
                        valid = False
                else:
                    valid = False
            else:
                valid = False
        
        if not valid:
            continue
        
        # Sort to maintain canonical form for comparison? Not strictly necessary for eval, but good for consistency
        trial_sparse.sort()
        
        # Check for duplicate exponents (should be handled above, but double check)
        exps = [x[0] for x in trial_sparse]
        if len(exps) != len(set(exps)):
            continue
            
        trial_dense = _sparse_to_dense(trial_sparse)
        
        # Quick validation
        if len(trial_dense) > 41 or trial_dense[-1] == 0:
            continue
        if sum(1 for c in trial_dense if c != 0) != 12:
            continue
        if any(c < -64 or c > 64 for c in trial_dense):
            continue
            
        trial_sq = _count_nonzero_square_fast(trial_dense)
        
        # Acceptance
        delta = trial_sq - current_sq
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
            current_sparse = trial_sparse
            current_dense = trial_dense
            current_sq = trial_sq
            stall_count = 0
            
            if trial_sq < best_sq:
                best_sq = trial_sq
                best_sparse = list(trial_sparse)
                # If we hit the theoretical minimum, we can stop early
                if best_sq == 12:
                    return best_sparse, best_sq
        else:
            stall_count += 1
            if stall_count > max_stall:
                # Restart from best
                current_sparse = list(best_sparse)
                current_dense = _sparse_to_dense(current_sparse)
                current_sq = best_sq
                stall_count = 0
                
    return best_sparse, best_sq

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin for 60s limit
    
    best_sparse = None
    best_sq = float('inf')
    
    # Phase 1: Quick Brute Force for Consecutive Exponents 0..11
    # This is very fast and often finds good local minima
    if time.time() < deadline - 10.0:
        # Only try if we haven't found a 12-term solution yet
        # 2^12 = 4096 iterations, very fast
        for bits in range(1 << 12):
            coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
            dense = coeffs
            if not _validate(dense):
                continue
            sq = _count_nonzero_square_fast(dense)
            if sq < best_sq:
                best_sq = sq
                best_sparse = _dense_to_sparse(dense)
                if best_sq == 12:
                    break
    
    # Phase 2: Generate diverse initial candidates
    candidates = []
    
    # 1. Consecutive exponents with random small coeffs
    random.seed(123)
    for _ in range(10):
        exps = list(range(12))
        coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        sparse = list(zip(exps, coeffs))
        candidates.append(sparse)
        
    # 2. Arithmetic Progressions
    for k in [1, 2, 3]:
        for start in [0, 1, 2]:
            exps = [start + i * k for i in range(12)]
            if exps[-1] > 40:
                continue
            coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
            sparse = list(zip(exps, coeffs))
            candidates.append(sparse)
            
    # 3. Random Exponents
    for _ in range(20):
        exps = sorted(random.sample(range(41), 12))
        coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
        sparse = list(zip(exps, coeffs))
        candidates.append(sparse)
        
    # 4. If we found something in Phase 1, include it
    if best_sparse:
        candidates.append(best_sparse)
        
    # Phase 3: Run Local Search on Candidates
    # Allocate time to each candidate
    n_candidates = len(candidates)
    time_per_candidate = (deadline - time.time() - 2.0) / n_candidates
    time_per_candidate = max(1.0, time_per_candidate) # Min 1s per candidate
    
    for i, cand in enumerate(candidates):
        if time.time() >= deadline - 2.0:
            break
        
        # Ensure validity before search
        dense = _sparse_to_dense(cand)
        if not _validate(dense):
            continue
            
        cand_sq = _count_nonzero_square_fast(dense)
        if cand_sq < best_sq:
            best_sq = cand_sq
            best_sparse = list(cand)
            if best_sq == 12:
                break
                
        # Run local search
        end_time = time.time() + time_per_candidate
        # Cap end time to global deadline
        if end_time > deadline - 2.0:
            end_time = deadline - 2.0
            
        found_sparse, found_sq = _local_search(cand, end_time, seed=42 + i)
        
        if found_sq < best_sq:
            best_sq = found_sq
            best_sparse = list(found_sparse)
            if best_sq == 12:
                break
                
    # Final Fallback
    if best_sparse is None:
        best_coeffs = [1] * 12
    else:
        best_coeffs = _sparse_to_dense(best_sparse)
        if not _validate(best_coeffs):
            best_coeffs = [1] * 12
            
    return best_coeffs
