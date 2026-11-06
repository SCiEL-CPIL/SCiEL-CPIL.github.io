"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def _count_nonzero(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _square_poly(coeffs):
    n = len(coeffs)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ai = coeffs[i]
        if ai == 0:
            continue
        for j in range(n):
            aj = coeffs[j]
            if aj == 0:
                continue
            sq[i + j] += ai * aj
    return sq

def _valid_candidate(coeffs):
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

def construct():
    deadline = time.time() + 80.0
    best_score = 100
    best_coeffs = None

    # Strategy: search over arithmetic progression exponents with binary signs
    # and small coefficient magnitudes to maximize collision density.
    
    # First, try pure binary signs with arithmetic progression exponents
    for d in range(1, 15):
        if time.time() > deadline:
            break
        # Degree range: exponents from 0 to d*11
        for base in range(0, 13):
            if time.time() > deadline:
                break
            exponents = [base + d * i for i in range(12)]
            max_deg = exponents[-1]
            if max_deg >= 41:
                continue
            for sign_pattern in range(4096):
                if time.time() > deadline:
                    break
                signs = [1 if (sign_pattern >> i) & 1 else -1 for i in range(12)]
                coeffs = [0] * (max_deg + 1)
                for i in range(12):
                    coeffs[exponents[i]] = signs[i]
                if not _valid_candidate(coeffs):
                    continue
                sq = _square_poly(coeffs)
                nz = _count_nonzero(sq)
                if nz < best_score:
                    best_score = nz
                    best_coeffs = coeffs[:]
    
    # If we found something good, try to improve with local search
    if best_coeffs is not None and best_score < 22:
        if time.time() < deadline - 5:
            # Local search: try small perturbations
            current = best_coeffs[:]
            current_nz = best_score
            for _ in range(50):
                if time.time() > deadline - 2:
                    break
                improved = False
                # Try changing a sign
                for i in range(len(current)):
                    if current[i] != 0:
                        old_val = current[i]
                        current[i] = -old_val
                        sq = _square_poly(current)
                        nz = _count_nonzero(sq)
                        if nz < current_nz:
                            current_nz = nz
                            best_score = nz
                            best_coeffs = current[:]
                            improved = True
                        else:
                            current[i] = old_val
                
                # Try moving a term to nearby degree
                for i in range(len(current)):
                    if current[i] != 0:
                        val = current[i]
                        for delta in [-2, -1, 1, 2]:
                            new_pos = i + delta
                            if 0 <= new_pos < 41 and current[new_pos] == 0:
                                current[i] = 0
                                current[new_pos] = val
                                if _valid_candidate(current):
                                    sq = _square_poly(current)
                                    nz = _count_nonzero(sq)
                                    if nz < current_nz:
                                        current_nz = nz
                                        best_score = nz
                                        best_coeffs = current[:]
                                        improved = True
                                current[new_pos] = 0
                                current[i] = val
    
    # Also try some specific known good patterns
    # Try patterns with coefficients of magnitude 1 or 2
    for d in range(1, 8):
        if time.time() > deadline:
            break
        for base in range(0, 13):
            if time.time() > deadline:
                break
            exponents = [base + d * i for i in range(12)]
            max_deg = exponents[-1]
            if max_deg >= 41:
                continue
            # Try all combinations of signs and magnitudes in {1, 2}
            # Too many combinations, so sample
            import random
            for _ in range(200):
                if time.time() > deadline:
                    break
                coeffs = [0] * (max_deg + 1)
                for i in range(12):
                    mag = random.choice([1, 1, 2])
                    sign = random.choice([1, -1])
                    coeffs[exponents[i]] = mag * sign
                if not _valid_candidate(coeffs):
                    continue
                sq = _square_poly(coeffs)
                nz = _count_nonzero(sq)
                if nz < best_score:
                    best_score = nz
                    best_coeffs = coeffs[:]

    if best_coeffs is None:
        # Fallback: seed program
        best_coeffs = [1] * 12
        best_score = 22
    
    return best_coeffs
# EVOLVE-BLOCK-END
