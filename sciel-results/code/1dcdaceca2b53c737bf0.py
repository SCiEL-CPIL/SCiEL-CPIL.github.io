"""Sparse squares of twelve-term integer polynomials."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _square_nonzero_count(coeffs):
    """Compute the number of nonzero terms in the exact square of a polynomial.
    coeffs is a list of integers (ascending degree)."""
    n = len(coeffs)
    # Use numpy convolution for exact integer arithmetic (small enough)
    c = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(c, c)
    # Trim trailing zeros but keep the structure
    # Count nonzero entries
    return int(np.count_nonzero(sq))

def _square_coeffs(coeffs):
    """Return the coefficient list of the square (ascending degree)."""
    n = len(coeffs)
    c = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(c, c)
    # Trim to remove trailing zeros for compactness, but keep at least 1
    # Actually we need to return it properly
    return sq.tolist()

def _validate(coeffs):
    """Check if coeffs satisfies all constraints."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    nonzero = sum(1 for c in coeffs if c != 0)
    if nonzero != 12:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    return True

def _score(coeffs):
    """Lower is better: number of nonzero terms in the square."""
    if not _validate(coeffs):
        return 999999
    return _square_nonzero_count(coeffs)

def _random_candidate(rng):
    """Generate a random 12-term polynomial with coefficients in [-64,64]."""
    # Choose 12 distinct positions in [0, 40]
    positions = sorted(rng.sample(range(41), 12))
    coeffs = [0] * 41
    for p in positions:
        coeffs[p] = rng.randint(-64, 64)
        if coeffs[p] == 0:
            coeffs[p] = 1
    # Trim trailing zeros
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    # Ensure last is nonzero
    if coeffs[-1] == 0:
        coeffs[-1] = 1
    return coeffs

def _local_search(coeffs, rng, deadline, best_score):
    """Local search: try perturbing individual coefficients and positions."""
    current = list(coeffs)
    current_score = _score(current)
    
    while time.time() < deadline:
        improved = False
        n = len(current)
        
        # Try changing each coefficient by small amounts
        for i in range(n):
            if time.time() >= deadline:
                break
            old_val = current[i]
            for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10]:
                new_val = old_val + delta
                if new_val < -64 or new_val > 64:
                    continue
                if new_val == 0 and old_val != 0:
                    # This would reduce nonzero count, check if still 12
                    pass
                current[i] = new_val
                # Check validity quickly
                nonzero = sum(1 for c in current if c != 0)
                if nonzero != 12:
                    current[i] = old_val
                    continue
                if len(current) > 41:
                    current[i] = old_val
                    continue
                if current[-1] == 0:
                    current[i] = old_val
                    continue
                s = _square_nonzero_count(current)
                if s < current_score:
                    current_score = s
                    best_score = min(best_score, s)
                    improved = True
                else:
                    current[i] = old_val
        
        # Try moving a nonzero term to a new position
        if time.time() >= deadline:
            break
        nonzero_indices = [i for i in range(n) if current[i] != 0]
        for idx in nonzero_indices:
            if time.time() >= deadline:
                break
            val = current[idx]
            # Try moving to a random new position
            for _ in range(5):
                new_pos = rng.randint(0, 40)
                if new_pos == idx:
                    continue
                # Check if new position is free
                if len(current) <= new_pos:
                    # Extend
                    current.extend([0] * (new_pos - len(current) + 1))
                if current[new_pos] != 0:
                    continue
                current[idx] = 0
                current[new_pos] = val
                # Trim
                while len(current) > 1 and current[-1] == 0:
                    current.pop()
                nonzero = sum(1 for c in current if c != 0)
                if nonzero != 12 or current[-1] == 0:
                    current[new_pos] = 0
                    current[idx] = val
                    # Restore length
                    while len(current) < max(n, idx+1, new_pos+1):
                        current.append(0)
                    continue
                s = _square_nonzero_count(current)
                if s < current_score:
                    current_score = s
                    best_score = min(best_score, s)
                    improved = True
                    break
                else:
                    current[new_pos] = 0
                    current[idx] = val
                    while len(current) < max(n, idx+1):
                        current.append(0)
        
        if not improved:
            # Try a larger random perturbation
            if time.time() >= deadline:
                break
            # Random restart within neighborhood
            new_coeffs = list(current)
            for _ in range(3):
                i = rng.randint(0, len(new_coeffs)-1)
                new_coeffs[i] = rng.randint(-64, 64)
                if new_coeffs[i] == 0:
                    new_coeffs[i] = 1
            # Ensure 12 nonzero
            nonzero = sum(1 for c in new_coeffs if c != 0)
            if nonzero != 12:
                # Fix
                if nonzero > 12:
                    nz = [i for i in range(len(new_coeffs)) if new_coeffs[i] != 0]
                    for i in nz[12:]:
                        new_coeffs[i] = 0
                elif nonzero < 12:
                    while sum(1 for c in new_coeffs if c != 0) < 12:
                        i = rng.randint(0, 40)
                        if len(new_coeffs) <= i:
                            new_coeffs.extend([0] * (i - len(new_coeffs) + 1))
                        if new_coeffs[i] == 0:
                            new_coeffs[i] = rng.randint(-64, 64)
                            if new_coeffs[i] == 0:
                                new_coeffs[i] = 1
            while len(new_coeffs) > 1 and new_coeffs[-1] == 0:
                new_coeffs.pop()
            if new_coeffs[-1] == 0:
                new_coeffs[-1] = 1
            s = _score(new_coeffs)
            if s < current_score:
                current = new_coeffs
                current_score = s
                best_score = min(best_score, s)
                improved = True
    
    return current, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    rng = random.Random(42)
    
    # Start with known good candidates
    candidates = []
    
    # Candidate 1: [1]*12 (the seed)
    candidates.append([1]*12)
    
    # Candidate 2: alternating signs
    candidates.append([1 if i%2==0 else -1 for i in range(12)])
    
    # Candidate 3: sparse with gaps
    c3 = [0]*25
    c3[0]=1; c3[3]=1; c3[6]=1; c3[9]=1; c3[12]=1; c3[15]=1
    c3[18]=1; c3[21]=1; c3[24]=1
    # Need 12 terms
    c3[1]=1; c3[4]=1; c3[7]=1; c3[10]=1
    candidates.append(c3)
    
    # Candidate 4: use structure where square is sparse
    # P = x^0 + x^2 + x^4 + ... + x^22 (12 terms, even powers)
    # P^2 = sum of x^(2i+2j) for i,j in 0..11
    # = x^0 + 2x^2 + 3x^4 + ... + 12x^22 + ... + x^44
    # This has 23 nonzero terms - not great
    # But with signs we can cancel
    c4 = [0]*45
    for i in range(12):
        c4[2*i] = 1 if i % 2 == 0 else -1
    candidates.append(c4)
    
    # Candidate 5: try to construct P where P^2 has few terms
    # If P = a + b where a^2 and b^2 are sparse and 2ab cancels
    # Try P with terms at positions that create many collisions in sums
    # Arithmetic progression: positions 0,1,2,...,11
    # Sums: 0,1,2,...,22 -> 23 distinct values
    # But coefficients can cancel if we choose signs wisely
    
    # Let's try a more structured approach:
    # P = 1 - x + x^2 - x^3 + ... + x^11 (alternating)
    # P = (1-x^12)/(1-x)
    # P^2 = (1-x^12)^2/(1-x)^2 = (1 - 2x^12 + x^24)/(1 - 2x + x^2)
    # This is a geometric-like series... let me compute
    # Actually (1-x^12)^2 = 1 - 2x^12 + x^24
    # Divided by (1-x)^2 = 1 - 2x + x^2
    # The result has terms from x^0 to x^24, many nonzero
    
    # Better: try to find P where P^2 is a perfect square of a sparse poly
    # If P^2 = Q^2 where Q is sparse, then P = ±Q
    # But P must have 12 terms, so Q must have 12 terms
    # And Q^2 should be sparse
    
    # Let me try a completely different approach:
    # Use the fact that (a+b+c+d)^2 can have cancellations
    # Split P into groups
    
    # For now, do random search + local search
    best_coeffs = [1]*12
    best_score = _score(best_coeffs)
    
    # Random search phase
    random_phase_deadline = start_time + 20
    while time.time() < random_phase_deadline:
        c = _random_candidate(rng)
        s = _score(c)
        if s < best_score:
            best_score = s
            best_coeffs = c
    
    # Local search from best
    best_coeffs, best_score = _local_search(best_coeffs, rng, deadline, best_score)
    
    # More random restarts with local search
    while time.time() < deadline:
        c = _random_candidate(rng)
        s = _score(c)
        if s < best_score:
            best_score = s
            best_coeffs = c
            c, best_score = _local_search(c, rng, deadline, best_score)
        else:
            # Short local search
            c, best_score = _local_search(c, rng, min(deadline, time.time()+5), best_score)
    
    # Ensure validity
    if not _validate(best_coeffs):
        best_coeffs = [1]*12
    
    return best_coeffs

# EVOLVE-BLOCK-END
