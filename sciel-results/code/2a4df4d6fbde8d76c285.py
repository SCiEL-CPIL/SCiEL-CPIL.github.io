"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _count_nonzero_square(coeffs):
    """Count nonzero terms in the exact square of a polynomial given by coefficient list."""
    n = len(coeffs)
    # Square convolution
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(n):
            if coeffs[j] == 0:
                continue
            sq[i + j] += coeffs[i] * coeffs[j]
    return sum(1 for c in sq if c != 0)

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

def _try_local_search(initial, deadline):
    """Local search: flip signs, swap exponents, small perturbations."""
    best_coeffs = initial[:]
    best_sq_count = _count_nonzero_square(initial)
    
    # Convert to sparse representation: list of (exponent, coeff)
    def to_sparse(coeffs):
        return [(i, c) for i, c in enumerate(coeffs) if c != 0]
    
    def from_sparse(sparse, max_deg=40):
        coeffs = [0] * (max_deg + 1)
        for exp, c in sparse:
            if 0 <= exp <= max_deg:
                coeffs[exp] = c
        # Trim trailing zeros but keep at least one
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs
    
    def sparse_to_list(sparse):
        if not sparse:
            return [0]
        max_exp = max(e for e, _ in sparse)
        coeffs = [0] * (max_exp + 1)
        for e, c in sparse:
            coeffs[e] = c
        return coeffs
    
    current_sparse = to_sparse(initial)
    current_list = sparse_to_list(current_sparse)
    current_sq = _count_nonzero_square(current_list)
    
    if current_sq < best_sq_count:
        best_sq_count = current_sq
        best_coeffs = current_list[:]
    
    random.seed(42)
    
    while time.time() < deadline - 2.0:
        # Pick a random move
        move_type = random.choice(['flip_sign', 'shift_exp', 'change_coeff', 'swap_terms'])
        
        trial_sparse = [(e, c) for e, c in current_sparse]
        
        if move_type == 'flip_sign':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            if c != 0:
                trial_sparse[idx] = (e, -c)
        
        elif move_type == 'shift_exp':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            shift = random.choice([-2, -1, 1, 2])
            new_e = e + shift
            if 0 <= new_e <= 40:
                # Remove old, add new
                trial_sparse.pop(idx)
                trial_sparse.append((new_e, c))
                # Check for duplicate exponents
                exps = [x[0] for x in trial_sparse]
                if len(exps) != len(set(exps)):
                    continue
                # Ensure we still have 12 terms
                if len(trial_sparse) != 12:
                    continue
        
        elif move_type == 'change_coeff':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            new_c = random.randint(-64, 64)
            if new_c != 0:
                trial_sparse[idx] = (e, new_c)
        
        elif move_type == 'swap_terms':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                trial_sparse[i] = (ej, ci)
                trial_sparse[j] = (ei, cj)
                exps = [x[0] for x in trial_sparse]
                if len(exps) != len(set(exps)):
                    continue
        
        # Evaluate
        trial_list = sparse_to_list(trial_sparse)
        if not _validate(trial_list):
            continue
        
        trial_sq = _count_nonzero_square(trial_list)
        
        if trial_sq < current_sq:
            current_sparse = trial_sparse
            current_list = trial_list
            current_sq = trial_sq
            if trial_sq < best_sq_count:
                best_sq_count = trial_sq
                best_coeffs = trial_list[:]
        # Accept with some probability to avoid getting stuck (simulated annealing-like)
        elif random.random() < 0.1:
            current_sparse = trial_sparse
            current_list = trial_list
            current_sq = trial_sq
    
    return best_coeffs, best_sq_count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial candidate: the seed
    initial = [1] * 12
    initial_sq = _count_nonzero_square(initial)
    best_coeffs = initial[:]
    best_sq = initial_sq
    
    # Try local search
    try:
        found_coeffs, found_sq = _try_local_search(initial, deadline)
        if found_sq < best_sq:
            best_coeffs = found_coeffs
            best_sq = found_sq
    except Exception:
        pass
    
    # Final validation
    if not _validate(best_coeffs):
        # Fallback to seed
        best_coeffs = [1] * 12
    
    return best_coeffs
# EVOLVE-BLOCK-END
