import time
import random
import sys

def _square_nonzero_count(coeffs):
    """Compute the square of a polynomial given as dict {exponent: coeff} and return nonzero count."""
    n = len(coeffs)
    if n == 0:
        return 0
    # Build list of (exp, coeff)
    items = list(coeffs.items())
    # Use dict for square
    sq = {}
    for i in range(n):
        ei, ci = items[i]
        for j in range(i, n):
            ej, cj = items[j]
            s = ei + ej
            if i == j:
                sq[s] = sq.get(s, 0) + ci * ci
            else:
                sq[s] = sq.get(s, 0) + 2 * ci * cj
    return sum(1 for v in sq.values() if v != 0)

def _square_dict(coeffs):
    """Return the square as a dict {exponent: coefficient}."""
    n = len(coeffs)
    items = list(coeffs.items())
    sq = {}
    for i in range(n):
        ei, ci = items[i]
        for j in range(i, n):
            ej, cj = items[j]
            s = ei + ej
            if i == j:
                sq[s] = sq.get(s, 0) + ci * ci
            else:
                sq[s] = sq.get(s, 0) + 2 * ci * cj
    return sq

def _count_nonzero(sq_dict):
    return sum(1 for v in sq_dict.values() if v != 0)

def _make_list_from_dict(d, max_deg=40):
    """Convert dict to coefficient list in ascending degree."""
    if not d:
        return [0]
    max_e = max(d.keys())
    lst = [0] * (max_e + 1)
    for e, c in d.items():
        if e <= max_deg:
            lst[e] = c
    # Trim trailing zeros but keep at least 1 element
    while len(lst) > 1 and lst[-1] == 0:
        lst.pop()
    return lst

def _validate(coeffs_dict):
    """Check if the polynomial satisfies all constraints."""
    if len(coeffs_dict) != 12:
        return False
    for e, c in coeffs_dict.items():
        if c == 0:
            return False
        if c < -64 or c > 64:
            return False
        if e < 0 or e > 40:
            return False
    max_e = max(coeffs_dict.keys())
    if max_e > 40:
        return False
    if coeffs_dict[max_e] == 0:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    best_score = None
    best_poly = None
    
    # Initial candidate: a structured polynomial
    # Try: terms at positions that create many sum collisions
    # Use exponents in two groups with overlap in sums
    
    random.seed(42)
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: alternating signs in arithmetic progression
    for sign_pattern in [1, -1]:
        exponents = list(range(0, 24, 2))  # 0,2,4,...,22
        coeffs = {}
        for i, e in enumerate(exponents):
            c = sign_pattern * (1 if i % 2 == 0 else -1)
            coeffs[e] = c
        candidates.append(coeffs.copy())
    
    # Candidate 2: random
    for _ in range(50):
        exponents = sorted(random.sample(range(0, 41), 12))
        coeffs = {}
        for e in exponents:
            c = random.choice([-2, -1, 1, 2])
            coeffs[e] = c
        candidates.append(coeffs)
    
    # Candidate 3: structured - pairs
    for _ in range(50):
        # Use 6 pairs of exponents
        base = random.sample(range(0, 30), 6)
        offsets = random.sample(range(1, 15), 6)
        exponents = []
        for i in range(6):
            exponents.append(base[i])
            exponents.append(base[i] + offsets[i])
        exponents = sorted(set(exponents))
        if len(exponents) != 12:
            continue
        coeffs = {}
        for i, e in enumerate(exponents):
            c = random.choice([-2, -1, 1, 2])
            coeffs[e] = c
        candidates.append(coeffs)
    
    # Evaluate initial candidates
    for cand in candidates:
        if not _validate(cand):
            continue
        sq = _square_dict(cand)
        score = _count_nonzero(sq)
        if best_score is None or score < best_score:
            best_score = score
            best_poly = cand.copy()
    
    if best_poly is None:
        # Fallback
        best_poly = {i: 1 for i in range(12)}
        best_score = _count_nonzero(_square_dict(best_poly))
    
    # Local search
    def get_score(poly):
        sq = _square_dict(poly)
        return _count_nonzero(sq)
    
    # Simulated annealing
    T = 2.0
    cooling = 0.9995
    
    current = best_poly.copy()
    current_score = best_score
    best_found = best_poly.copy()
    best_found_score = best_score
    
    step = 0
    no_improve = 0
    
    while time.time() < deadline:
        step += 1
        
        # Choose a move
        move_type = random.random()
        new_poly = current.copy()
        
        if move_type < 0.3:
            # Change a coefficient
            idx = random.randrange(12)
            exps = list(new_poly.keys())
            e = exps[idx]
            old_c = new_poly[e]
            # Try new coefficient
            new_c = random.choice([-64, -32, -16, -8, -4, -3, -2, -1, 1, 2, 3, 4, 8, 16, 32, 64])
            if new_c == old_c:
                new_c = old_c + 1
            if new_c == 0:
                new_c = 1
            new_poly[e] = new_c
        elif move_type < 0.5:
            # Change an exponent
            idx = random.randrange(12)
            exps = list(new_poly.keys())
            old_e = exps[idx]
            c = new_poly[old_e]
            new_e = random.randint(0, 40)
            if new_e in new_poly:
                continue
            del new_poly[old_e]
            new_poly[new_e] = c
        elif move_type < 0.6:
            # Swap two exponents
            i, j = random.sample(range(12), 2)
            exps = list(new_poly.keys())
            e1, e2 = exps[i], exps[j]
            c1, c2 = new_poly[e1], new_poly[e2]
            if e1 == e2:
                continue
            del new_poly[e1]
            del new_poly[e2]
            new_poly[e2] = c1
            new_poly[e1] = c2
        elif move_type < 0.7:
            # Change sign of a coefficient
            idx = random.randrange(12)
            exps = list(new_poly.keys())
            e = exps[idx]
            new_poly[e] = -new_poly[e]
        elif move_type < 0.8:
            # Add a small perturbation to a coefficient
            idx = random.randrange(12)
            exps = list(new_poly.keys())
            e = exps[idx]
            old_c = new_poly[e]
            delta = random.choice([-2, -1, 1, 2])
            new_c = old_c + delta
            if new_c == 0 or abs(new_c) > 64:
                new_c = old_c + 1
                if new_c == 0 or abs(new_c) > 64:
                    new_c = old_c - 1
                    if new_c == 0 or abs(new_c) > 64:
                        continue
            new_poly[e] = new_c
        else:
            # Move an exponent by a small amount
            idx = random.randrange(12)
            exps = list(new_poly.keys())
            old_e = exps[idx]
            c = new_poly[old_e]
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_e = old_e + shift
            if new_e < 0 or new_e > 40 or new_e in new_poly:
                continue
            del new_poly[old_e]
            new_poly[new_e] = c
        
        if not _validate(new_poly):
            continue
        
        new_score = get_score(new_poly)
        
        # Accept or reject
        if new_score <= current_score:
            current = new_poly
            current_score = new_score
            if new_score < best_found_score:
                best_found = new_poly.copy()
                best_found_score = new_score
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Simulated annealing acceptance
            delta = new_score - current_score
            if random.random() < (1.0 / (1.0 + delta * T * 0.5)):
                current = new_poly
                current_score = new_score
            no_improve += 1
        
        T *= cooling
        
        if no_improve > 5000:
            # Restart with best found
            current = best_found.copy()
            current_score = best_found_score
            no_improve = 0
            T = max(T, 1.0)
    
    # Final phase: exhaustive sign search on best polynomial
    if time.time() < deadline - 5:
        best_exps = sorted(best_found.keys())
        best_coeffs = [best_found[e] for e in best_exps]
        
        # Try all sign combinations (2^12 = 4096)
        from itertools import product
        for signs in product([1, -1], repeat=12):
            if time.time() > deadline - 2:
                break
            trial = {}
            for i, e in enumerate(best_exps):
                c = abs(best_coeffs[i]) * signs[i]
                if c != 0:
                    trial[e] = c
            if not _validate(trial):
                continue
            sq = _square_dict(trial)
            score = _count_nonzero(sq)
            if score < best_found_score:
                best_found = trial
                best_found_score = score
    
    # Also try scaling: multiply all by 2 or -1 doesn't help, but try coefficient variations
    if time.time() < deadline - 3:
        # Try changing one coefficient at a time in best polynomial
        for i in range(12):
            if time.time() > deadline - 2:
                break
            e = best_exps[i]
            old_c = best_found[e]
            for new_c in range(-64, 65):
                if new_c == 0 or new_c == old_c:
                    continue
                trial = best_found.copy()
                trial[e] = new_c
                if not _validate(trial):
                    continue
                sq = _square_dict(trial)
                score = _count_nonzero(sq)
                if score < best_found_score:
                    best_found = trial
                    best_found_score = score
    
    # Convert to list
    result = _make_list_from_dict(best_found, max_deg=40)
    
    # Final validation
    if len(result) > 41:
        result = result[:41]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    # Count nonzero terms
    nz = sum(1 for c in result if c != 0)
    if nz != 12:
        # Fallback
        result = [1] * 12
    
    return result

