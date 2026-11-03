import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
_START_TIME = time.time()
DEADLINE = _START_TIME + 110  # Safety margin before 120s limit

def _time_remaining():
    return DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _to_half_int_list(val):
    """Convert a value to a half-integer if possible, else return None."""
    if isinstance(val, Fraction):
        if val.denominator == 1:
            return val.numerator
        elif val.denominator == 2:
            return val.numerator
        else:
            return None
    elif isinstance(val, float):
        doubled = val * 2
        rounded = round(doubled)
        if abs(doubled - rounded) < 1e-9:
            return rounded
        return None
    elif isinstance(val, int):
        return val * 2
    return None

def _solve_for_W(U, V, n, m, p, r):
    """
    Given U (n*m x r) and V (m*p x r), solve for W (p*n x r)
    such that for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    For each (i,k), we have m equations in r unknowns:
    sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for j=0..m-1
    """
    W = [[0]*r for _ in range(p*n)]
    
    for i in range(n):
        for k in range(p):
            # Build the m x r matrix A_ik where A_ik[j,c] = U[i*m+j,c]*V[j*p+k,c]
            # We need to find w (length r) such that A_ik @ w = ones(m)
            
            # Try to find a sparse solution with small support
            # First try single term
            found = False
            for c in range(r):
                vals = [U[i*m+j][c] * V[j*p+k][c] for j in range(m)]
                if all(v == vals[0] for v in vals) and vals[0] != 0:
                    w = Fraction(1, vals[0])
                    hw = _to_half_int_list(w)
                    if hw is not None:
                        W[k*n+i][c] = hw
                        found = True
                        break
            
            if found:
                continue
            
            # Try two-term solutions
            for c1 in range(r):
                if found:
                    break
                for c2 in range(c1+1, r):
                    a_vals = [U[i*m+j][c1] * V[j*p+k][c1] for j in range(m)]
                    b_vals = [U[i*m+j][c2] * V[j*p+k][c2] for j in range(m)]
                    
                    # Solve using first two equations if m >= 2
                    if m >= 2:
                        a0, b0 = a_vals[0], b_vals[0]
                        a1, b1 = a_vals[1], b_vals[1]
                        det = a0*b1 - a1*b0
                        if det == 0:
                            continue
                        
                        w1 = Fraction(b1 - b0, det)
                        w2 = Fraction(a0 - a1, det)
                        
                        valid = True
                        for j in range(m):
                            check = a_vals[j]*w1 + b_vals[j]*w2
                            if check != 1:
                                valid = False
                                break
                        
                        if valid:
                            h1 = _to_half_int_list(w1)
                            h2 = _to_half_int_list(w2)
                            if h1 is not None and h2 is not None:
                                W[k*n+i][c1] = h1
                                W[k*n+i][c2] = h2
                                found = True
                                break
                    else:
                        # m == 1
                        a0 = a_vals[0]
                        b0 = b_vals[0]
                        if a0 != 0 and b0 != 0:
                            # Try small half-integer values
                            for w1_half in range(-20, 21, 2):
                                w1 = Fraction(w1_half, 2)
                                w2 = (1 - a0*w1) / b0
                                h2 = _to_half_int_list(w2)
                                if h2 is not None:
                                    W[k*n+i][c1] = w1_half
                                    W[k*n+i][c2] = h2
                                    found = True
                                    break
                            if found:
                                break
            
            if not found:
                # Try three-term solutions
                for c1 in range(r):
                    if found:
                        break
                    for c2 in range(c1+1, r):
                        if found:
                            break
                        for c3 in range(c2+1, r):
                            a_vals = [U[i*m+j][c1] * V[j*p+k][c1] for j in range(m)]
                            b_vals = [U[i*m+j][c2] * V[j*p+k][c2] for j in range(m)]
                            d_vals = [U[i*m+j][c3] * V[j*p+k][c3] for j in range(m)]
                            
                            # Solve 3x3 system
                            if m >= 3:
                                # Use Cramer's rule
                                det = (a_vals[0]*(b_vals[1]*d_vals[2] - b_vals[2]*d_vals[1])
                                     - b_vals[0]*(a_vals[1]*d_vals[2] - a_vals[2]*d_vals[1])
                                     + d_vals[0]*(a_vals[1]*b_vals[2] - a_vals[2]*b_vals[1]))
                                if det == 0:
                                    continue
                                
                                w1_num = (1*(b_vals[1]*d_vals[2] - b_vals[2]*d_vals[1])
                                         - b_vals[0]*(1*d_vals[2] - a_vals[2]*1)
                                         + d_vals[0]*(a_vals[1]*1 - a_vals[2]*1))
                                w2_num = (a_vals[0]*(1*d_vals[2] - d_vals[1]*1)
                                         - 1*(a_vals[1]*d_vals[2] - a_vals[2]*d_vals[1])
                                         + d_vals[0]*(a_vals[1]*1 - a_vals[2]*1))
                                w3_num = (a_vals[0]*(b_vals[1]*1 - b_vals[2]*1)
                                         - b_vals[0]*(a_vals[1]*1 - a_vals[2]*1)
                                         + 1*(a_vals[1]*b_vals[2] - a_vals[2]*b_vals[1]))
                                
                                w1 = Fraction(w1_num, det)
                                w2 = Fraction(w2_num, det)
                                w3 = Fraction(w3_num, det)
                                
                                valid = True
                                for j in range(m):
                                    check = a_vals[j]*w1 + b_vals[j]*w2 + d_vals[j]*w3
                                    if check != 1:
                                        valid = False
                                        break
                                
                                if valid:
                                    h1 = _to_half_int_list(w1)
                                    h2 = _to_half_int_list(w2)
                                    h3 = _to_half_int_list(w3)
                                    if h1 is not None and h2 is not None and h3 is not None:
                                        W[k*n+i][c1] = h1
                                        W[k*n+i][c2] = h2
                                        W[k*n+i][c3] = h3
                                        found = True
                                        break
                            elif m == 2:
                                # Underdetermined, try to find a solution
                                # a0*w1 + b0*w2 + d0*w3 = 1
                                # a1*w1 + b1*w2 + d1*w3 = 1
                                # Try small values for w1
                                for w1_half in range(-10, 11, 2):
                                    w1 = Fraction(w1_half, 2)
                                    # Solve 2x2 for w2, w3
                                    # b0*w2 + d0*w3 = 1 - a0*w1
                                    # b1*w2 + d1*w3 = 1 - a1*w1
                                    rhs0 = 1 - a_vals[0]*w1
                                    rhs1 = 1 - a_vals[1]*w1
                                    det2 = b_vals[0]*d_vals[1] - b_vals[1]*d_vals[0]
                                    if det2 == 0:
                                        continue
                                    w2 = (rhs0*d_vals[1] - rhs1*d_vals[0]) / det2
                                    w3 = (b_vals[0]*rhs1 - b_vals[1]*rhs0) / det2
                                    h2 = _to_half_int_list(w2)
                                    h3 = _to_half_int_list(w3)
                                    if h2 is not None and h3 is not None:
                                        W[k*n+i][c1] = w1_half
                                        W[k*n+i][c2] = h2
                                        W[k*n+i][c3] = h3
                                        found = True
                                        break
                                if found:
                                    break
            
            if not found:
                return None
    
    return W

def _try_random_pattern(n, m, p, r, seed, density=0.3):
    """Try a random binary pattern for U and V, then solve for W."""
    rng = random.Random(seed)
    
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    
    for i in range(n*m):
        for c in range(r):
            if rng.random() < density:
                U[i][c] = rng.choice([-1, 1])
    
    for j in range(m*p):
        for c in range(r):
            if rng.random() < density:
                V[j][c] = rng.choice([-1, 1])
    
    W = _solve_for_W(U, V, n, m, p, r)
    if W is None:
        return None
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Lower bound is max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    for target_rank in range(12, best_rank):
        if _time_remaining() < 5:
            break
        
        max_attempts = 50 if target_rank <= 20 else 30
        if _time_remaining() < 15:
            max_attempts = 10
        
        for attempt in range(max_attempts):
            if _time_remaining() < 2:
                break
            
            seed = attempt * 7919 + 12345
            
            result = _try_random_pattern(n, m, p, target_rank, seed)
            if result is not None:
                best = result
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            continue
    
    return best

import random
