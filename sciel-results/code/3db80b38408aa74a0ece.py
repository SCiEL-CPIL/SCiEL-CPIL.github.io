import time
import random

_DEADLINE = 90.0
_START = None

def _nonzero_count(poly):
    return sum(1 for c in poly if c != 0)

def _square_support(coeffs):
    """Return number of nonzero terms in the exact integer square of coeffs."""
    n = len(coeffs)
    # Find nonzero indices
    nz_indices = [i for i, c in enumerate(coeffs) if c != 0]
    if not nz_indices:
        return 0
    
    # Dictionary for square coefficients
    sq = {}
    for i in nz_indices:
        a = coeffs[i]
        for j in nz_indices:
            b = coeffs[j]
            if b == 0: continue
            k = i + j
            sq[k] = sq.get(k, 0) + a * b
            
    return sum(1 for v in sq.values() if v != 0)

def _valid(candidate):
    if not isinstance(candidate, list):
        return False
    if len(candidate) == 0 or len(candidate) > 41:
        return False
    if candidate[-1] == 0:
        return False
    for c in candidate:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
    if sum(1 for c in candidate if c != 0) != 12:
        return False
    return True

def _make_poly_from_exponents(exponents, coeffs):
    # exponents and coeffs are lists of length 12
    # return list of length max(exponents)+1
    max_e = max(exponents)
    poly = [0] * (max_e + 1)
    for e, c in zip(exponents, coeffs):
        poly[e] = c
    return poly

def _local_search(coeffs, rng, deadline):
    best = _square_support(coeffs)
    best_c = list(coeffs)
    n = len(best_c)
    max_degree = 40
    
    # Coefficients to try
    coeff_candidates = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 8, -8, 16, -16, 32, -32, 64, -64]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try changing coefficients
        nz_indices = [i for i in range(n) if best_c[i] != 0]
        for i in nz_indices:
            for new_c in coeff_candidates:
                if new_c == best_c[i]:
                    continue
                trial = list(best_c)
                trial[i] = new_c
                s = _square_support(trial)
                if s < best:
                    best = s
                    best_c = trial
                    improved = True
                elif s == best and rng.random() < 0.05:
                    best_c = trial
                    improved = True
        
        # Try moving a term
        for i in nz_indices:
            if best_c[i] == 0: continue
            # Try moving to nearby positions
            for delta in range(-3, 4):
                if delta == 0: continue
                j = i + delta
                if 0 <= j < max_degree + 1 and best_c[j] == 0:
                    trial = list(best_c)
                    # Extend if needed
                    if j >= len(trial):
                        trial.extend([0] * (j - len(trial) + 1))
                    trial[j] = trial[i]
                    trial[i] = 0
                    # Trim trailing zeros to keep length <= 41
                    while len(trial) > 1 and trial[-1] == 0:
                        trial.pop()
                    if len(trial) > 41:
                        continue
                    if trial[-1] == 0 and len(trial) > 1:
                        # Ensure nonzero final coeff? The task says nonzero final coefficient.
                        # If we move the highest term down, we must ensure the new highest is nonzero.
                        # trial[j] is nonzero, so if j is the new max, it's fine.
                        # But if we moved a lower term, the old max might become 0 if it was the only one?
                        # No, we only move one term. If i was the max, and we move it to j < i, 
                        # then trial[i] becomes 0. The new max is max(j, other max).
                        # We need to ensure the polynomial list ends with a nonzero.
                        # The _valid check handles this.
                        pass
                    
                    s = _square_support(trial)
                    if s < best:
                        best = s
                        best_c = trial
                        improved = True
                    elif s == best and rng.random() < 0.05:
                        best_c = trial
                        improved = True
                        
            # Try moving to a random position
            if rng.random() < 0.1:
                j = rng.randint(0, max_degree)
                if j != i and best_c[j] == 0:
                    trial = list(best_c)
                    if j >= len(trial):
                        trial.extend([0] * (j - len(trial) + 1))
                    trial[j] = trial[i]
                    trial[i] = 0
                    while len(trial) > 1 and trial[-1] == 0:
                        trial.pop()
                    if len(trial) <= 41:
                        s = _square_support(trial)
                        if s < best:
                            best = s
                            best_c = trial
                            improved = True
                            
        # Update nz_indices if best_c changed
        if improved:
            # Re-evaluate nz_indices in next loop iteration implicitly
            pass

    return best_c, best

def construct():
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + _DEADLINE - 2.0
    
    rng = random.Random(42)
    
    best_c = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    
    # 1. Arithmetic progression 0..11, alternating signs
    c1 = [0]*12
    for i in range(12):
        c1[i] = 1 if i % 2 == 0 else -1
    candidates.append(c1)
    
    # 2. Arithmetic progression 0..11, all 1s
    c2 = [1]*12
    candidates.append(c2)
    
    # 3. Arithmetic progression 0,2,4..22, alternating
    c3 = [0]*23
    for i in range(12):
        c3[2*i] = 1 if i % 2 == 0 else -1
    candidates.append(c3)
    
    # 4. Random
    for _ in range(20):
        exponents = sorted(rng.sample(range(41), 12))
        coeffs = [rng.choice([1, -1, 2, -2, 3, -3]) for _ in range(12)]
        # Ensure last is nonzero
        if coeffs[-1] == 0: coeffs[-1] = 1
        poly = _make_poly_from_exponents(exponents, coeffs)
        candidates.append(poly)
        
    # Evaluate initial candidates
    for cand in candidates:
        if not _valid(cand):
            continue
        s = _square_support(cand)
        if s < best_score:
            best_score = s
            best_c = list(cand)
            
    # Local search on best and some random restarts
    if best_c:
        best_c, best_score = _local_search(best_c, rng, deadline)
        
    # Random restarts
    restarts = 50
    for i in range(restarts):
        if time.time() > deadline:
            break
            
        # Generate random candidate
        degree = rng.randint(11, 40)
        exponents = sorted(rng.sample(range(degree + 1), 12))
        coeffs = [rng.choice([1, -1, 2, -2, 3, -3, 4, -4]) for _ in range(12)]
        if coeffs[-1] == 0: coeffs[-1] = 1
        cand = _make_poly_from_exponents(exponents, coeffs)
        
        if not _valid(cand):
            continue
            
        s = _square_support(cand)
        if s < best_score:
            best_score = s
            best_c = list(cand)
            
        # Local search if promising
        if s <= best_score + 2:
            cand2, s2 = _local_search(cand, rng, deadline)
            if s2 < best_score:
                best_score = s2
                best_c = cand2
                
    # Final polish
    if best_c and time.time() < deadline:
        best_c, best_score = _local_search(best_c, rng, deadline)
        
    if not best_c:
        # Fallback
        best_c = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        
    # Ensure validity
    if not _valid(best_c):
        best_c = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
        
    return best_c
