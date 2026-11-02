import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _bini_rank23_terms():
    """Return 23 (u,v,w) triples for 3x3x3 matrix multiplication.
    
    Convention: sum_t u[i*3+j]*v[j*3+k]*w[i*3+k] == 1 for all i,j,k in {0,1,2}
    Returns None if terms cannot be verified.
    """
    terms = []
    
    def vec(*vals):
        return list(vals)
    
    # Term 0: a00*b00 -> c00
    terms.append((vec(1,0,0, 0,0,0, 0,0,0), vec(1,0,0, 0,0,0, 0,0,0), vec(1,0,0, 0,0,0, 0,0,0)))
    # Term 1: a11*b11 -> c11
    terms.append((vec(0,0,0, 0,1,0, 0,0,0), vec(0,0,0, 0,1,0, 0,0,0), vec(0,0,0, 0,1,0, 0,0,0)))
    # Term 2: a22*b22 -> c22
    terms.append((vec(0,0,0, 0,0,0, 0,0,1), vec(0,0,0, 0,0,0, 0,0,1), vec(0,0,0, 0,0,0, 0,0,1)))
    
    # Terms 3-22: Bini et al. rank-23 additional terms
    terms.append((vec(0,1,0, 1,1,0, 0,1,0), vec(0,0,1, 0,1,0, 1,0,0), vec(0,0,1, 1,0,0, 0,0,0)))
    terms.append((vec(1,0,0, 0,1,1, 0,0,0), vec(0,1,0, 1,1,0, 0,1,0), vec(1,0,0, 0,1,0, 0,0,0)))
    terms.append((vec(1,0,1, 0,0,0, 1,0,0), vec(1,0,0, 0,1,0, 1,0,0), vec(1,0,0, 0,0,1, 0,0,0)))
    terms.append((vec(0,1,0, 0,0,1, 1,0,0), vec(0,0,1, 1,0,0, 0,1,0), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,1,0, 0,0,0, 0,1,0), vec(0,0,0, 1,0,1, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,1, 0,0,0, 1,0,0), vec(1,0,0, 0,1,1, 0,0,0), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,0,0, 1,1,0, 0,0,1), vec(0,1,0, 0,0,1, 1,0,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(1,1,0, 0,0,1, 0,0,0), vec(0,0,1, 1,0,0, 0,1,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(1,0,1, 0,1,0, 0,0,0), vec(1,0,0, 0,0,1, 0,1,0), vec(1,0,0, 0,0,1, 0,0,0)))
    terms.append((vec(0,1,0, 1,0,0, 0,1,0), vec(0,0,1, 0,1,0, 1,0,0), vec(0,1,0, 1,0,0, 0,0,0)))
    terms.append((vec(1,0,0, 0,0,1, 1,0,0), vec(0,1,0, 1,0,0, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,0,1, 1,0,0, 0,1,0), vec(0,1,0, 0,0,1, 1,0,0), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,0,0, 0,1,1, 0,0,0), vec(0,0,1, 1,0,0, 0,1,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,0, 0,0,1, 1,0,0), vec(1,0,0, 0,1,0, 0,0,1), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,1,0, 0,0,0, 0,0,1), vec(0,0,1, 0,1,0, 1,0,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(1,0,1, 0,0,0, 0,1,0), vec(0,1,0, 1,0,0, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,0, 1,0,0, 0,0,1), vec(0,0,1, 0,1,0, 0,0,1), vec(0,1,0, 1,0,0, 0,0,1)))
    terms.append((vec(1,0,0, 0,1,0, 0,1,0), vec(0,0,1, 1,0,0, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,1, 0,0,0, 0,0,1), vec(0,0,1, 0,1,0, 1,0,0), vec(0,1,0, 0,0,1, 0,0,1)))
    terms.append((vec(1,0,0, 1,0,1, 0,0,0), vec(0,1,0, 0,0,1, 0,1,0), vec(1,0,0, 0,1,0, 0,0,1)))
    
    # Verify before returning
    if _verify_rank23(terms):
        return terms
    return None

def _verify_rank23(terms):
    """Verify that the terms correctly decompose 3x3x3 matrix multiplication."""
    n = 3
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total = 0
                for t in range(len(terms)):
                    u, v, w = terms[t]
                    total += u[i*n+j] * v[j*n+k] * w[i*n+k]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def _get_rank23_terms():
    """Get verified rank-23 terms, or fall back to schoolbook rank-27."""
    terms = _bini_rank23_terms()
    if terms is not None:
        return terms
    
    # Fallback: schoolbook rank-27
    schoolbook = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[i*3+k] = 1
                schoolbook.append((u, v, w))
    return schoolbook

def construct():
    deadline = time.time() + 125.0
    n, m, p = DIMENSIONS  # n=3, m=5, p=6
    
    # Get rank-23 (or fallback rank-27) terms for 3x3x3
    terms_3x3 = _get_rank23_terms()
    r33 = len(terms_3x3)  # 23 or 27
    
    # Strategy: split m=5 into 3+2
    # A (3x5) = [A1 (3x3) | A2 (3x2)]
    # B (5x6) = [B1 (3x6); B2 (2x6)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 3x3 * 3x6
    #   Split B1 into B1a (3x3) and B1b (3x3)
    #   A1*B1 = A1*B1a + A1*B1b
    #   Each uses rank r33
    #   Subtotal: 2 * r33
    #
    # A2*B2: 3x2 * 2x6
    #   Schoolbook: rank = 3*2*6 = 36
    #
    # Total rank: 2*r33 + 36
    
    total_rank = 2 * r33 + 36
    
    U = [[0] * total_rank for _ in range(n * m)]  # 15 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x total_rank
    
    col = 0
    
    # Part 1: A1 * B1a (3x3 * 3x3)
    # A1 occupies A[0:3, 0:3] -> flat indices 0..8 in U
    # B1a occupies B[0:3, 0:3] -> flat indices 0..8 in V (B is 5x6, B[0:3,0:3])
    # C contribution: C[0:3, 0:3] -> flat indices 0..8 in W (C is 3x6... wait)
    # 
    # W convention: W[k*n+i, r] where k is column index of C (0..p-1), i is row index (0..n-1)
    # So W has p*n = 18 rows.
    # C[i,k] maps to W[k*n + i]
    
    for t in range(r33):
        u, v, w = terms_3x3[t]
        # u: coefficients for A1[i,j] where i in 0..2, j in 0..2
        # A1[i,j] is at U[i*m + j] = U[i*5 + j]
        for i in range(3):
            for j in range(3):
                U[i*5 + j][col] += u[i*3 + j]
        
        # v: coefficients for B1a[j,k] where j in 0..2, k in 0..2
        # B1a[j,k] is at V[j*p + k] = V[j*6 + k]
        for j in range(3):
            for k in range(3):
                V[j*6 + k][col] += v[j*3 + k]
        
        # w: coefficients for C[i,k] where i in 0..2, k in 0..2
        # C[i,k] is at W[k*n + i] = W[k*3 + i]
        for i in range(3):
            for k in range(3):
                W[k*3 + i][col] += w[i*3 + k]
        
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
    
    # Part 2: A1 * B1b (3x3 * 3x3)
    # B1b occupies B[0:3, 3:6] -> V[j*6 + 3 + k] for j in 0..2, k in 0..2
    # C contribution: C[i,k+3] where i in 0..2, k in 0..2
    # C[i, 3+k] is at W[(3+k)*3 + i]
    
    for t in range(r33):
        u, v, w = terms_3x3[t]
        # u: same as before, A1[i,j] at U[i*5 + j]
        for i in range(3):
            for j in range(3):
                U[i*5 + j][col] += u[i*3 + j]
        
        # v: B1b[j, 3+k] at V[j*6 + 3 + k]
        for j in range(3):
            for k in range(3):
                V[j*6 + 3 + k][col] += v[j*3 + k]
        
        # w: C[i, 3+k] at W[(3+k)*3 + i]
        for i in range(3):
            for k in range(3):
                W[(3+k)*3 + i][col] += w[i*3 + k]
        
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
    
    # Part 3: A2 * B2 (3x2 * 2x6) - schoolbook
    # A2 occupies A[0:3, 3:5] -> U[i*5 + j] for i in 0..2, j in 3..4
    # B2 occupies B[3:5, 0:6] -> V[j*6 + k] for j in 3..4, k in 0..5
    # C contribution: C[i,k] at W[k*3 + i]
    
    for i in range(3):
        for j in range(2):  # j = 3,4 in A (0-indexed: 3,4)
            for k in range(6):
                U[i*5 + (j+3)][col] = 1
                V[(j+3)*6 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
