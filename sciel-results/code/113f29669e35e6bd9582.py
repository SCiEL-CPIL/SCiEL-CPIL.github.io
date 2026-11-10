import random
import time
import math

def count_nonzero(lst):
    return sum(1 for x in lst if x != 0)

def poly_square(coeffs):
    """Compute exact square of polynomial given as coeff list (ascending degree)."""
    n = len(coeffs)
    if n == 0:
        return []
    result = [0] * (2*n - 1)
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    for k, (i, ci) in enumerate(nz):
        for l in range(k, len(nz)):
            j, cj = nz[l]
            val = ci * cj
            if i == j:
                result[i+j] += val
            else:
                result[i+j] += 2 * val
    return result

def valid_candidate(coeffs):
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    if count_nonzero(coeffs) != 12:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    return True

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    best_score = float('inf')
    best_cand = None
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Structured patterns
    # Pattern: 1, 0, ..., 1 (sparse)
    for _ in range(20):
        exps = sorted(random.sample(range(41), 12))
        p = [0]*41
        for e in exps:
            p[e] = random.choice([1, -1, 2, -2, 3, -3])
        while p and p[-1] == 0:
            p.pop()
        if valid_candidate(p):
            candidates.append(p)
            
    # Pattern: Clumped
    for _ in range(20):
        base = random.randint(0, 29)
        exps = sorted(set(base + random.randint(0, 11) for _ in range(12)))
        if len(exps) < 12:
             # Fill gaps
             while len(exps) < 12:
                 exps.append(random.randint(0, 40))
             exps = sorted(set(exps))[:12]
             if len(exps) < 12: continue
        p = [0]*41
        for e in exps:
            p[e] = random.choice([1, -1])
        while p and p[-1] == 0:
            p.pop()
        if valid_candidate(p):
            candidates.append(p)

    # 2. Known good structures if any (e.g. from math)
    # (1+x+x^2+x^3+x^4+x^5+x^6)^2 has many terms.
    # (1-x+x^2-x^3+x^4-x^5+x^6)^2?
    # Let's just rely on the search.

    if not candidates:
        # Fallback
        p = [0]*41
        for i in range(0, 41, 3):
            p[i] = 1
        # Ensure 12 terms
        nz_count = count_nonzero(p)
        if nz_count < 12:
            for i in range(nz_count, 12):
                 p[random.randint(0,40)] = 1
        while p and p[-1] == 0:
            p.pop()
        if valid_candidate(p):
            candidates.append(p)
        else:
            # Hardcode a valid one
            p = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
            # This has 14 terms. Need exactly 12.
            # Let's just use a simple valid one
            p = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # Count: 2 terms. Need 12.
            for i in range(12):
                 p[i*3] = 1
            p = p[:41]
            while p and p[-1] == 0:
                p.pop()
            candidates.append(p)

    # Simulated Annealing
    for start_cand in candidates:
        if time.time() - start_time > time_limit:
            break
            
        current = start_cand[:]
        # Ensure length 41 for internal ops, but valid_candidate trims
        # We work with fixed length 41 internally to simplify indexing, then trim for validation
        curr_full = [0]*41
        for i, c in enumerate(current):
            curr_full[i] = c
            
        curr_score = count_nonzero(poly_square(curr_full))
        
        if curr_score < best_score:
            best_score = curr_score
            best_cand = curr_full[:]
            
        T = 100.0
        T_min = 0.1
        cooling_rate = 0.995
        
        steps = 0
        max_steps = 50000
        
        while T > T_min and steps < max_steps:
            if time.time() - start_time > time_limit:
                break
                
            steps += 1
            neighbor_full = curr_full[:]
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Change value of a nonzero term
                nz_idx = [i for i, c in enumerate(neighbor_full) if c != 0]
                if nz_idx:
                    idx = random.choice(nz_idx)
                    neighbor_full[idx] = random.randint(-64, 64)
                    if neighbor_full[idx] == 0:
                        neighbor_full[idx] = 1
            elif move_type < 0.8:
                # Move a nonzero term to a new position
                nz_idx = [i for i, c in enumerate(neighbor_full) if c != 0]
                if nz_idx:
                    idx = random.choice(nz_idx)
                    val = neighbor_full[idx]
                    neighbor_full[idx] = 0
                    new_idx = random.randint(0, 40)
                    if neighbor_full[new_idx] == 0:
                        neighbor_full[new_idx] = val
                    else:
                        # If collision, just put it there and let validity check fail?
                        # No, we need to maintain 12 nonzero terms.
                        # If we move to occupied slot, we lose a term.
                        # So we must swap or pick empty.
                        # Let's find an empty slot
                        empty_slots = [i for i in range(41) if neighbor_full[i] == 0]
                        if empty_slots:
                            new_idx = random.choice(empty_slots)
                            neighbor_full[new_idx] = val
                        else:
                            # No empty slots? Impossible if we have 12 terms in 41 slots
                            neighbor_full[idx] = val
            else:
                # Swap values of two nonzero terms
                nz_idx = [i for i, c in enumerate(neighbor_full) if c != 0]
                if len(nz_idx) >= 2:
                    i1, i2 = random.sample(nz_idx, 2)
                    neighbor_full[i1], neighbor_full[i2] = neighbor_full[i2], neighbor_full[i1]
            
            # Validate
            # Trim for validation
            nb_trimmed = neighbor_full[:]
            while nb_trimmed and nb_trimmed[-1] == 0:
                nb_trimmed.pop()
                
            if not valid_candidate(nb_trimmed):
                T *= cooling_rate
                continue
                
            nb_score = count_nonzero(poly_square(neighbor_full))
            
            delta = nb_score - curr_score
            if delta < 0:
                curr_score = nb_score
                curr_full = neighbor_full
                if curr_score < best_score:
                    best_score = curr_score
                    best_cand = curr_full[:]
            else:
                if random.random() < math.exp(-delta / T):
                    curr_score = nb_score
                    curr_full = neighbor_full
                    
            T *= cooling_rate
            
    # Final cleanup
    if best_cand is None:
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
    # Ensure final return is trimmed and valid
    while best_cand and best_cand[-1] == 0:
        best_cand.pop()
    if not valid_candidate(best_cand):
        # Fallback
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
    return best_cand
