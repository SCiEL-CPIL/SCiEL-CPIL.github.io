import random
import time
from collections import Counter

def _square_nonzero_count(coeffs):
    """Compute number of nonzero terms in the exact square of polynomial given by coeff list."""
    n = len(coeffs)
    # Square: c[k] = sum_{i+j=k} coeffs[i]*coeffs[j]
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(i, n):
            if coeffs[j] == 0:
                continue
            val = coeffs[i] * coeffs[j]
            if i == j:
                sq[i + j] += val
            else:
                sq[i + j] += 2 * val
    # Trim trailing zeros
    while len(sq) > 1 and sq[-1] == 0:
        sq.pop()
    return sum(1 for c in sq if c != 0)

def _is_valid(coeffs):
    """Check all constraints."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    nz = sum(1 for c in coeffs if c != 0)
    if nz != 12:
        return False
    for c in coeffs:
        if not isinstance(c, int) or c < -64 or c > 64:
            return False
    return True

def _try_local_search(init_coeffs, max_steps=50000, deadline=None):
    """Hill-climbing: try changing one coefficient or moving a term."""
    best = init_coeffs[:]
    best_count = _square_nonzero_count(best)
    n = len(best)
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        # Try modifying a random coefficient
        idx = random.randrange(n)
        old_val = best[idx]
        # Choose new value: small integers or 0
        new_val = random.choice([0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 8, -8, 16, -16, 32, -32])
        if new_val == old_val:
            continue
        if new_val != 0 and (new_val < -64 or new_val > 64):
            continue
        best[idx] = new_val
        # Check validity
        if not _is_valid(best):
            best[idx] = old_val
            continue
        cnt = _square_nonzero_count(best)
        if cnt < best_count:
            best_count = cnt
        else:
            best[idx] = old_val
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    best_coeffs = [1] * 12
    best_count = _square_nonzero_count(best_coeffs)  # 78
    
    # Try alternating signs
    alt = [1 if i % 2 == 0 else -1 for i in range(12)]
    if _is_valid(alt):
        c = _square_nonzero_count(alt)
        if c < best_count:
            best_count = c
            best_coeffs = alt[:]
    
    # Random restarts with local search
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Generate random 12-term polynomial
        # Choose 12 distinct positions from 0..40
        positions = random.sample(range(41), 12)
        positions.sort()
        coeffs = [0] * 41
        for pos in positions:
            # Random coefficient from small set
            coeffs[pos] = random.choice([1, -1, 2, -2, 3, -3, 4, -4, 5, -5])
        
        if not _is_valid(coeffs):
            continue
        
        # Local search
        cand, cnt = _try_local_search(coeffs, max_steps=3000, deadline=deadline)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = cand[:]
        
        # Also try structured patterns
        if restart % 10 == 0:
            # Try patterns with specific structure
            for pattern_type in range(5):
                if time.time() > deadline:
                    break
                p = [0] * 41
                if pattern_type == 0:
                    # Symmetric: a_i = a_{n-1-i}
                    for i in range(6):
                        v = random.choice([1, -1, 2, -2])
                        p[i] = v
                        p[11 - i] = v
                elif pattern_type == 1:
                    # Antisymmetric
                    for i in range(6):
                        v = random.choice([1, -1, 2, -2])
                        p[i] = v
                        p[11 - i] = -v
                elif pattern_type == 2:
                    # Sparse: only even positions
                    for i in range(6):
                        p[2 * i] = random.choice([1, -1, 2, -2])
                elif pattern_type == 3:
                    # Two clusters
                    for i in range(6):
                        p[i] = random.choice([1, -1, 2, -2])
                    for i in range(6):
                        p[20 + i] = random.choice([1, -1, 2, -2])
                elif pattern_type == 4:
                    # Geometric progression of exponents
                    base = random.randint(1, 3)
                    for i in range(12):
                        pos = i * base
                        if pos < 41:
                            p[pos] = random.choice([1, -1])
                
                if _is_valid(p):
                    cand, cnt = _try_local_search(p, max_steps=2000, deadline=deadline)
                    if cnt < best_count:
                        best_count = cnt
                        best_coeffs = cand[:]
    
    # Trim trailing zeros
    while len(best_coeffs) > 1 and best_coeffs[-1] == 0:
        best_coeffs.pop()
    
    return best_coeffs
