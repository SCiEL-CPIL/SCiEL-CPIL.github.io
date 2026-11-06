import time
import random
import numpy as np
from itertools import product

def _square_nonzero_fast(exps, coeffs):
    """Compute nonzero count of square using vectorized numpy."""
    n = len(exps)
    if n == 0:
        return 0
    e = np.array(exps, dtype=np.int32)
    c = np.array(coeffs, dtype=np.int32)
    # Compute all pairwise sums and products
    # For i<=j: contribution to degree e[i]+e[j]
    # Use outer sum trick
    ei = e[:, np.newaxis]
    ej = e[np.newaxis, :]
    sums = ei + ej
    ci = c[:, np.newaxis]
    cj = c[np.newaxis, :]
    prods = ci * cj
    # For i<j, count twice (since both (i,j) and (j,i) appear in outer product)
    # For i==j, count once
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    diag = np.eye(n, dtype=bool)
    # Build full contribution: for upper triangle use 2*ci*cj, for diagonal use ci^2
    contrib = np.zeros((n, n), dtype=np.int64)
    contrib[diag] = prods[diag]
    contrib[mask_upper] = 2 * prods[mask_upper]
    contrib[np.tril(np.ones((n, n), dtype=bool), k=-1)] = 0  # lower triangle not used
    
    # Now accumulate by degree
    max_deg = int(sums.max()) if n > 0 else 0
    sq = np.zeros(max_deg + 1, dtype=np.int64)
    for i in range(n):
        for j in range(i, n):
            s = sums[i, j]
            sq[s] += contrib[i, j]
    return int(np.sum(sq != 0))

def _square_nonzero_fast2(exps, coeffs):
    """Faster: use numpy bincount for degree accumulation."""
    n = len(exps)
    if n == 0:
        return 0
    e = np.array(exps, dtype=np.int32)
    c = np.array(coeffs, dtype=np.int32)
    
    # Build all pairs (i, j) with i <= j
    i_idx, j_idx = np.triu_indices(n, k=0)
    degrees = e[i_idx] + e[j_idx]
    # Products: for i==j it's c[i]^2, for i!=j it's 2*c[i]*c[j]
    prods = c[i_idx] * c[j_idx]
    diag_mask = (i_idx == j_idx)
    prods[~diag_mask] *= 2
    
    # Accumulate using bincount
    max_deg = int(degrees.max()) if len(degrees) > 0 else 0
    sq = np.bincount(degrees, weights=prods, minlength=max_deg + 1).astype(np.int64)
    return int(np.sum(sq != 0))

def _make_list_from_exps_coeffs(exps, coeffs):
    """Convert parallel lists to coefficient list in ascending degree."""
    max_e = max(exps) if exps else 0
    lst = [0] * (max_e + 1)
    for e, c in zip(exps, coeffs):
        if e <= 40:
            lst[e] = c
    while len(lst) > 1 and lst[-1] == 0:
        lst.pop()
    return lst

def _validate(exps, coeffs):
    """Check validity."""
    if len(exps) != 12:
        return False
    seen = set()
    for e, c in zip(exps, coeffs):
        if c == 0 or c < -64 or c > 64:
            return False
        if e < 0 or e > 40:
            return False
        if e in seen:
            return False
        seen.add(e)
    return True

def _hill_climb(exps, coeffs, deadline, rng):
    """Perform hill climbing from initial state, return best found."""
    current_exps = exps[:]
    current_coeffs = coeffs[:]
    current_score = _square_nonzero_fast2(current_exps, current_coeffs)
    best_exps = current_exps[:]
    best_coeffs = current_coeffs[:]
    best_score = current_score
    
    no_improve = 0
    
    while time.time() < deadline:
        move_type = rng.random()
        new_exps = current_exps[:]
        new_coeffs = current_coeffs[:]
        
        if move_type < 0.30:
            # Change a coefficient
            idx = rng.randrange(12)
            new_c = rng.choice([-8, -4, -3, -2, -1, 1, 2, 3, 4, 8])
            if new_c == new_coeffs[idx]:
                new_c = new_coeffs[idx] + 1 if new_coeffs[idx] < 64 else new_coeffs[idx] - 1
            if new_c == 0:
                new_c = 1
            new_coeffs[idx] = new_c
        elif move_type < 0.50:
            # Change an exponent
            idx = rng.randrange(12)
            new_e = rng.randint(0, 40)
            if new_e in new_exps:
                continue
            new_exps[idx] = new_e
            new_exps.sort()
        elif move_type < 0.65:
            # Change sign of a coefficient
            idx = rng.randrange(12)
            new_coeffs[idx] = -new_coeffs[idx]
        elif move_type < 0.80:
            # Move an exponent by small amount
            idx = rng.randrange(12)
            old_e = new_exps[idx]
            shift = rng.choice([-3, -2, -1, 1, 2, 3])
            new_e = old_e + shift
            if new_e < 0 or new_e > 40 or new_e in new_exps:
                continue
            new_exps[idx] = new_e
            new_exps.sort()
        elif move_type < 0.90:
            # Swap two exponents
            i, j = rng.sample(range(12), 2)
            new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
            new_exps.sort()
        else:
            # Change two coefficients
            i, j = rng.sample(range(12), 2)
            new_coeffs[i] = rng.choice([-4, -2, -1, 1, 2, 4])
            new_coeffs[j] = rng.choice([-4, -2, -1, 1, 2, 4])
        
        if not _validate(new_exps, new_coeffs):
            continue
        
        new_score = _square_nonzero_fast2(new_exps, new_coeffs)
        
        if new_score <= current_score:
            current_exps = new_exps
            current_coeffs = new_coeffs
            current_score = new_score
            if new_score < best_score:
                best_exps = new_exps[:]
                best_coeffs = new_coeffs[:]
                best_score = new_score
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        if no_improve > 500:
            break
    
    return best_exps, best_coeffs, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    rng = random.Random(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Structured families
    # 1. Consecutive
    candidates.append((list(range(12)), [1] * 12))
    # 2. Alternating signs
    candidates.append((list(range(12)), [1 if i % 2 == 0 else -1 for i in range(12)]))
    # 3. Even spacing
    candidates.append(([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], [1] * 12))
    # 4. Fibonacci-like spacing
    fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    fib = [min(f, 40) for f in fib]
    # Make unique
    seen = set()
    unique_fib = []
    for f in fib:
        if f not in seen and f <= 40:
            unique_fib.append(f)
            seen.add(f)
    while len(unique_fib) < 12:
        unique_fib.append(len(unique_fib) * 3)
    candidates.append((sorted(unique_fib[:12]), [1] * 12))
    # 5. Powers of 2
    p2 = [min(2**i, 40) for i in range(12)]
    seen = set()
    unique_p2 = []
    for p in p2:
        if p not in seen:
            unique_p2.append(p)
            seen.add(p)
    while len(unique_p2) < 12:
        unique_p2.append(len(unique_p2) + 1)
    candidates.append((sorted(unique_p2[:12]), [1] * 12))
    # 6. Dense at start
    candidates.append((list(range(0, 12)), [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]))
    # 7. Symmetric
    candidates.append(([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], [1]*12))
    
    # Random candidates
    for _ in range(30):
        exps = sorted(rng.sample(range(0, 41), 12))
        coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
        candidates.append((exps, coeffs))
    
    # Evaluate all initial candidates
    best_exps = None
    best_coeffs = None
    best_score = float('inf')
    
    for exps, coeffs in candidates:
        if not _validate(exps, coeffs):
            continue
        score = _square_nonzero_fast2(exps, coeffs)
        if score < best_score:
            best_score = score
            best_exps = exps[:]
            best_coeffs = coeffs[:]
    
    if best_exps is None:
        best_exps = list(range(12))
        best_coeffs = [1] * 12
        best_score = _square_nonzero_fast2(best_exps, best_coeffs)
    
    # Multi-restart hill climbing
    restarts = 20
    time_per_restart = (deadline - start_time) / (restarts + 1)
    
    for r in range(restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            # Use the best initial
            start_exps = best_exps[:]
            start_coeffs = best_coeffs[:]
        else:
            # Generate new random starting point
            start_exps = sorted(rng.sample(range(0, 41), 12))
            start_coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        local_deadline = min(start_time + (r + 1) * time_per_restart, deadline - 5)
        
        le, lc, ls = _hill_climb(start_exps, start_coeffs, local_deadline, rng)
        
        if ls < best_score:
            best_score = ls
            best_exps = le[:]
            best_coeffs = lc[:]
    
    # Final sign optimization on best exponents
    if time.time() < deadline - 3:
        abs_coeffs = [abs(best_coeffs[i]) for i in range(12)]
        for signs in product([1, -1], repeat=12):
            if time.time() > deadline - 2:
                break
            trial_coeffs = [abs_coeffs[i] * signs[i] for i in range(12)]
            if not _validate(best_exps, trial_coeffs):
                continue
            score = _square_nonzero_fast2(best_exps, trial_coeffs)
            if score < best_score:
                best_score = score
                best_coeffs = trial_coeffs[:]
    
    # Convert to list
    result = _make_list_from_exps_coeffs(best_exps, best_coeffs)
    
    if len(result) > 41:
        result = result[:41]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    nz = sum(1 for c in result if c != 0)
    if nz != 12:
        result = [1] * 12
    
    return result
