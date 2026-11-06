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

def _hill_climb_ap_fixed(exps, coeffs, deadline, rng):
    """Hill climbing with fixed exponents (AP), optimizing coefficients."""
    current_exps = exps[:]
    current_coeffs = coeffs[:]
    current_score = _square_nonzero_fast(current_exps, current_coeffs)
    best_exps = current_exps[:]
    best_coeffs = current_coeffs[:]
    best_score = current_score
    
    no_improve = 0
    max_iters = 5000
    
    while time.time() < deadline and no_improve < 1000:
        move_type = rng.random()
        new_exps = current_exps[:]
        new_coeffs = current_coeffs[:]
        
        if move_type < 0.5:
            # Change a coefficient
            idx = rng.randrange(12)
            # Try small integers
            new_c = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, -8, 8])
            if new_c == 0:
                new_c = 1
            new_coeffs[idx] = new_c
        elif move_type < 0.8:
            # Flip sign
            idx = rng.randrange(12)
            new_coeffs[idx] = -new_coeffs[idx]
        else:
            # Change two coefficients
            i, j = rng.sample(range(12), 2)
            new_coeffs[i] = rng.choice([-2, -1, 1, 2])
            new_coeffs[j] = rng.choice([-2, -1, 1, 2])
        
        if not _validate(new_exps, new_coeffs):
            continue
        
        new_score = _square_nonzero_fast(new_exps, new_coeffs)
        
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
            # Accept worse with small probability (simulated annealing lite)
            if rng.random() < 0.05:
                current_exps = new_exps
                current_coeffs = new_coeffs
                current_score = new_score
            no_improve += 1

    return best_exps, best_coeffs, best_score

def _hill_climb_general(exps, coeffs, deadline, rng):
    """General hill climbing allowing exponent changes."""
    current_exps = exps[:]
    current_coeffs = coeffs[:]
    current_score = _square_nonzero_fast(current_exps, current_coeffs)
    best_exps = current_exps[:]
    best_coeffs = current_coeffs[:]
    best_score = current_score
    
    no_improve = 0
    
    while time.time() < deadline and no_improve < 2000:
        move_type = rng.random()
        new_exps = current_exps[:]
        new_coeffs = current_coeffs[:]
        
        if move_type < 0.4:
            # Change coefficient
            idx = rng.randrange(12)
            new_c = rng.choice([-8, -4, -3, -2, -1, 1, 2, 3, 4, 8])
            if new_c == 0:
                new_c = 1
            new_coeffs[idx] = new_c
        elif move_type < 0.6:
            # Move exponent by small amount
            idx = rng.randrange(12)
            old_e = new_exps[idx]
            shift = rng.choice([-2, -1, 1, 2])
            new_e = old_e + shift
            if new_e < 0 or new_e > 40 or new_e in new_exps:
                continue
            new_exps[idx] = new_e
            new_exps.sort()
        elif move_type < 0.7:
            # Flip sign
            idx = rng.randrange(12)
            new_coeffs[idx] = -new_coeffs[idx]
        else:
            # Change two coefficients
            i, j = rng.sample(range(12), 2)
            new_coeffs[i] = rng.choice([-2, -1, 1, 2])
            new_coeffs[j] = rng.choice([-2, -1, 1, 2])
        
        if not _validate(new_exps, new_coeffs):
            continue
        
        new_score = _square_nonzero_fast(new_exps, new_coeffs)
        
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
            if rng.random() < 0.02:
                current_exps = new_exps
                current_coeffs = new_coeffs
                current_score = new_score
            no_improve += 1

    return best_exps, best_coeffs, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    rng = random.Random(42)
    
    # Initial candidates
    candidates = []
    
    # 1. AP with all 1s
    candidates.append((list(range(12)), [1] * 12))
    # 2. AP with alternating signs
    candidates.append((list(range(12)), [1 if i % 2 == 0 else -1 for i in range(12)]))
    # 3. AP with [1, -1, 1, -1 ...] starting with -1
    candidates.append((list(range(12)), [-1 if i % 2 == 0 else 1 for i in range(12)]))
    # 4. AP with random small coeffs
    for _ in range(5):
        coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
        candidates.append((list(range(12)), coeffs))
        
    # 5. Some random sparse exponents
    for _ in range(10):
        exps = sorted(rng.sample(range(0, 41), 12))
        coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
        candidates.append((exps, coeffs))
    
    # Evaluate initial candidates
    best_exps = None
    best_coeffs = None
    best_score = float('inf')
    
    for exps, coeffs in candidates:
        if not _validate(exps, coeffs):
            continue
        score = _square_nonzero_fast(exps, coeffs)
        if score < best_score:
            best_score = score
            best_exps = exps[:]
            best_coeffs = coeffs[:]
    
    if best_exps is None:
        best_exps = list(range(12))
        best_coeffs = [1] * 12
        best_score = _square_nonzero_fast(best_exps, best_coeffs)
    
    # Phase 1: Deep dive on the best AP-like candidate (fixed exponents)
    # We assume the best candidate has exponents close to AP.
    # Let's force exponents to 0..11 for the best coefficient set to see if it helps.
    # Actually, just run _hill_climb_ap_fixed on the current best.
    if time.time() < deadline - 20:
        e1, c1, s1 = _hill_climb_ap_fixed(best_exps, best_coeffs, deadline - 20, rng)
        if s1 < best_score:
            best_exps, best_coeffs, best_score = e1, c1, s1
    
    # Phase 2: General hill climbing from the best found so far
    if time.time() < deadline - 5:
        e2, c2, s2 = _hill_climb_general(best_exps, best_coeffs, deadline - 5, rng)
        if s2 < best_score:
            best_exps, best_coeffs, best_score = e2, c2, s2
            
    # Phase 3: Try a few random restarts of general hill climbing
    remaining = deadline - time.time()
    if remaining > 10:
        restarts = 3
        time_per_restart = remaining / (restarts + 1)
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
            start_exps = sorted(rng.sample(range(0, 41), 12))
            # Bias towards AP
            if rng.random() < 0.5:
                start_exps = list(range(12))
                # Perturb
                idx = rng.randrange(12)
                start_exps[idx] = rng.randint(0, 40)
                if start_exps[idx] in start_exps:
                    start_exps[idx] = (start_exps[idx] + 1) % 41
                    if start_exps[idx] in start_exps:
                        start_exps = list(range(12))
                start_exps.sort()
            start_coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
            
            local_deadline = min(start_time + (r + 1) * time_per_restart, deadline - 5)
            le, lc, ls = _hill_climb_general(start_exps, start_coeffs, local_deadline, rng)
            
            if ls < best_score:
                best_score = ls
                best_exps = le[:]
                best_coeffs = lc[:]

    # Final sign optimization on best exponents
    if time.time() < deadline - 2:
        abs_coeffs = [abs(best_coeffs[i]) for i in range(12)]
        for signs in product([1, -1], repeat=12):
            if time.time() > deadline - 1:
                break
            trial_coeffs = [abs_coeffs[i] * signs[i] for i in range(12)]
            if not _validate(best_exps, trial_coeffs):
                continue
            score = _square_nonzero_fast(best_exps, trial_coeffs)
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
