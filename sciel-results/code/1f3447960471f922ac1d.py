"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_nonzero_square_fast(coeffs):
    """Count nonzero terms in the exact square using NumPy convolution."""
    n = len(coeffs)
    if n == 0:
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
    # Trim trailing zeros but keep at least one
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs

def _dense_to_sparse(coeffs):
    """Convert dense coefficient list to sparse representation."""
    return [(i, c) for i, c in enumerate(coeffs) if c != 0]

def _local_search_numba_style(sparse_init, deadline):
    """
    Local search with NumPy-accelerated evaluation.
    Uses coefficient values from {-2,-1,1,2} and explores exponent shifts.
    """
    coeff_choices = [-2, -1, 1, 2]
    
    current_sparse = list(sparse_init)
    current_dense = _sparse_to_dense(current_sparse)
    current_sq = _count_nonzero_square_fast(current_dense)
    
    best_sparse = list(current_sparse)
    best_dense = current_dense[:]
    best_sq = current_sq
    
    random.seed(42)
    
    # Pre-compute valid exponent ranges
    max_exp = 40
    
    iteration = 0
    while time.time() < deadline - 3.0:
        iteration += 1
        
        # Choose move type with weighted probability
        move_type = random.choices(
            ['flip_sign', 'shift_exp', 'change_coeff', 'shift_two', 'coeff_two'],
            weights=[30, 30, 15, 15, 10]
        )[0]
        
        trial_sparse = list(current_sparse)
        valid = True
        
        if move_type == 'flip_sign':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            trial_sparse[idx] = (e, -c)
            
        elif move_type == 'shift_exp':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_e = e + shift
            if 0 <= new_e <= max_exp:
                # Check for collision
                existing_exps = set(x[0] for x in trial_sparse if x[0] != e)
                if new_e not in existing_exps:
                    trial_sparse[idx] = (new_e, c)
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'change_coeff':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            new_c = random.choice(coeff_choices)
            trial_sparse[idx] = (e, new_c)
            
        elif move_type == 'shift_two':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                # Shift one by small amount
                shift_i = random.choice([-2, -1, 1, 2])
                shift_j = random.choice([-2, -1, 1, 2])
                new_ei = ei + shift_i
                new_ej = ej + shift_j
                if 0 <= new_ei <= max_exp and 0 <= new_ej <= max_exp:
                    # Check collisions
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
                
        elif move_type == 'coeff_two':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                # Change two coefficients
                new_ci = random.choice(coeff_choices)
                new_cj = random.choice(coeff_choices)
                trial_sparse[i] = (ei, new_ci)
                trial_sparse[j] = (ej, new_cj)
            else:
                valid = False
        
        if not valid:
            continue
        
        # Ensure exactly 12 terms with distinct exponents
        if len(trial_sparse) != 12:
            continue
        exps = [x[0] for x in trial_sparse]
        if len(exps) != len(set(exps)):
            continue
        
        trial_dense = _sparse_to_dense(trial_sparse)
        if not _validate(trial_dense):
            continue
        
        trial_sq = _count_nonzero_square_fast(trial_dense)
        
        # Accept if better, or with small probability for exploration
        if trial_sq < current_sq:
            current_sparse = trial_sparse
            current_dense = trial_dense
            current_sq = trial_sq
            if trial_sq < best_sq:
                best_sq = trial_sq
                best_sparse = list(trial_sparse)
                best_dense = trial_dense[:]
        elif random.random() < 0.05:
            current_sparse = trial_sparse
            current_dense = trial_dense
            current_sq = trial_sq
    
    return best_dense, best_sq

def _systematic_search(deadline):
    """
    Systematic search over structured polynomial families.
    Try patterns that might yield sparse squares.
    """
    best_coeffs = None
    best_sq = float('inf')
    
    random.seed(123)
    
    # Try various structured patterns
    patterns = []
    
    # Pattern 1: Alternating signs
    for base in range(0, 12):
        coeffs = [0] * 41
        for i in range(12):
            exp = base + i * 3  # spread out
            if exp < 41:
                coeffs[exp] = 1 if i % 2 == 0 else -1
        if sum(1 for c in coeffs if c != 0) == 12 and coeffs[-1] != 0:
            # Trim
            while len(coeffs) > 1 and coeffs[-1] == 0:
                coeffs.pop()
            if _validate(coeffs):
                sq = _count_nonzero_square_fast(coeffs)
                if sq < best_sq:
                    best_sq = sq
                    best_coeffs = coeffs[:]
    
    # Pattern 2: Grouped terms with cancellation
    for group_size in [3, 4]:
        num_groups = 12 // group_size
        for start in range(0, 10):
            coeffs = [0] * 41
            valid = True
            for g in range(num_groups):
                base_exp = start + g * (group_size + 1)
                for k in range(group_size):
                    exp = base_exp + k
                    if exp >= 41:
                        valid = False
                        break
                    # Use pattern that might cancel
                    if k == 0:
                        coeffs[exp] = 1
                    elif k == group_size - 1:
                        coeffs[exp] = -1
                    else:
                        coeffs[exp] = 1 if random.random() < 0.5 else -1
                if not valid:
                    break
            if valid and sum(1 for c in coeffs if c != 0) == 12:
                while len(coeffs) > 1 and coeffs[-1] == 0:
                    coeffs.pop()
                if _validate(coeffs):
                    sq = _count_nonzero_square_fast(coeffs)
                    if sq < best_sq:
                        best_sq = sq
                        best_coeffs = coeffs[:]
    
    # Pattern 3: Random but structured
    for _ in range(200):
        coeffs = [0] * 41
        # Choose 12 random exponents
        exps = sorted(random.sample(range(0, 41), 12))
        for e in exps:
            coeffs[e] = random.choice([-2, -1, 1, 2])
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        if _validate(coeffs):
            sq = _count_nonzero_square_fast(coeffs)
            if sq < best_sq:
                best_sq = sq
                best_coeffs = coeffs[:]
    
    if best_coeffs is not None:
        return best_coeffs, best_sq
    return None, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Start with a good initial candidate
    # Use a pattern known to have some cancellation
    initial_sparse = [(i * 3, 1 if i % 2 == 0 else -1) for i in range(12)]
    # Filter to valid range
    initial_sparse = [(e, c) for e, c in initial_sparse if e < 41]
    # Pad to 12 if needed
    while len(initial_sparse) < 12:
        initial_sparse.append((len(initial_sparse) * 3, 1))
    initial_sparse = initial_sparse[:12]
    
    initial_dense = _sparse_to_dense(initial_sparse)
    if not _validate(initial_dense):
        # Fallback
        initial_dense = [1] * 12
        initial_sparse = _dense_to_sparse(initial_dense)
    
    best_coeffs = initial_dense[:]
    best_sq = _count_nonzero_square_fast(best_coeffs)
    
    # Phase 1: Systematic search
    try:
        sys_coeffs, sys_sq = _systematic_search(deadline - 30.0)
        if sys_coeffs is not None and sys_sq < best_sq:
            best_coeffs = sys_coeffs
            best_sq = sys_sq
    except Exception:
        pass
    
    # Phase 2: Local search from best
    if time.time() < deadline - 10.0:
        try:
            current_sparse = _dense_to_sparse(best_coeffs)
            found_coeffs, found_sq = _local_search_numba_style(current_sparse, deadline)
            if found_sq < best_sq:
                best_coeffs = found_coeffs
                best_sq = found_sq
        except Exception:
            pass
    
    # Final validation
    if not _validate(best_coeffs):
        best_coeffs = [1] * 12
    
    return best_coeffs
# EVOLVE-BLOCK-END
