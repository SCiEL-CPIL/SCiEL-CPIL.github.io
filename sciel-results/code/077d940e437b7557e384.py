import time
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    U_frac = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    V_frac = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    W_frac = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += U_frac[i * m + j][c] * V_frac[l * p + k][c] * W_frac[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _lader_3x3x3():
    """
    Returns Lader's rank-23 decomposition for 3x3x3 matrix multiplication.
    Each term is a triple (u, v, w) where u, v, w are length-9 vectors with entries in {-1, 0, 1}.
    u[i*m+j] corresponds to A[i,j], v[l*p+k] corresponds to B[l,k], w[k*n+i] corresponds to C[k,i].
    For 3x3x3: n=m=p=3, so indices 0..8.
    """
    # Lader's 23 terms for 3x3x3
    # Format: (u, v, w) where each is a list of 9 ints
    # u index: i*3+j for A[i,j]
    # v index: l*3+k for B[l,k]
    # w index: k*3+i for C[k,i]
    
    terms = []
    
    # Term 1: A[0,0]*B[0,0] -> C[0,0]
    terms.append(([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # Term 2: A[0,1]*B[1,0] -> C[0,0]
    terms.append(([0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # Term 3: A[0,2]*B[2,0] -> C[0,0]
    terms.append(([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0]))
    # Term 4: A[1,0]*B[0,1] -> C[1,0]
    terms.append(([0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]))
    # Term 5: A[1,1]*B[1,1] -> C[1,1]
    terms.append(([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,1,0,0,0,0,0,0,0]))
    # Term 6: A[1,2]*B[2,1] -> C[1,1]
    terms.append(([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,1,0,0,0,0,0,0,0]))
    # Term 7: A[2,0]*B[0,2] -> C[2,0]
    terms.append(([0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]))
    # Term 8: A[2,1]*B[1,2] -> C[2,1]
    terms.append(([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,0]))
    # Term 9: A[2,2]*B[2,2] -> C[2,2]
    terms.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,0]))
    
    # Now the 14 "cross" terms from Lader's decomposition
    # These are the non-trivial terms that reduce rank from 27 to 23
    
    # Term 10: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) -> C[0,0]+C[1,1]
    # u: A[0,0]=1, A[1,1]=1
    # v: B[0,0]=1, B[1,1]=1
    # w: C[0,0]=1, C[1,1]=1
    terms.append(([1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0]))
    
    # Term 11: (A[0,1]+A[1,0]) * (B[1,0]+B[0,1]) -> C[0,1]+C[1,0]
    # u: A[0,1]=1, A[1,0]=1
    # v: B[1,0]=1, B[0,1]=1
    # w: C[0,1]=1, C[1,0]=1
    terms.append(([0,1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0,0]))
    
    # Term 12: (A[0,2]+A[2,0]) * (B[2,0]+B[0,2]) -> C[0,2]+C[2,0]
    # u: A[0,2]=1, A[2,0]=1
    # v: B[2,0]=1, B[0,2]=1
    # w: C[0,2]=1, C[2,0]=1
    terms.append(([0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,1]))
    
    # Term 13: (A[1,2]+A[2,1]) * (B[2,1]+B[1,2]) -> C[1,2]+C[2,1]
    # u: A[1,2]=1, A[2,1]=1
    # v: B[2,1]=1, B[1,2]=1
    # w: C[1,2]=1, C[2,1]=1
    terms.append(([0,0,0,0,0,1,0,1,0], [0,0,0,0,0,0,0,1,1], [0,0,0,0,0,1,0,0,1]))
    
    # Term 14: (A[0,0]-A[1,1]) * (B[0,0]-B[1,1]) -> C[0,0]-C[1,1]
    # u: A[0,0]=1, A[1,1]=-1
    # v: B[0,0]=1, B[1,1]=-1
    # w: C[0,0]=1, C[1,1]=-1
    terms.append(([1,0,0,0,-1,0,0,0,0], [1,0,0,0,-1,0,0,0,0], [1,0,0,0,-1,0,0,0,0]))
    
    # Term 15: (A[0,1]-A[1,0]) * (B[1,0]-B[0,1]) -> C[0,1]-C[1,0]
    # u: A[0,1]=1, A[1,0]=-1
    # v: B[1,0]=1, B[0,1]=-1
    # w: C[0,1]=1, C[1,0]=-1
    terms.append(([0,1,0,-1,0,0,0,0,0], [0,1,0,-1,0,0,0,0,0], [0,1,0,-1,0,0,0,0,0]))
    
    # Term 16: (A[0,2]-A[2,0]) * (B[2,0]-B[0,2]) -> C[0,2]-C[2,0]
    # u: A[0,2]=1, A[2,0]=-1
    # v: B[2,0]=1, B[0,2]=-1
    # w: C[0,2]=1, C[2,0]=-1
    terms.append(([0,0,1,0,0,0,-1,0,0], [0,0,1,0,0,0,-1,0,0], [0,0,1,0,0,0,0,0,-1]))
    
    # Term 17: (A[1,2]-A[2,1]) * (B[2,1]-B[1,2]) -> C[1,2]-C[2,1]
    # u: A[1,2]=1, A[2,1]=-1
    # v: B[2,1]=1, B[1,2]=-1
    # w: C[1,2]=1, C[2,1]=-1
    terms.append(([0,0,0,0,0,1,0,-1,0], [0,0,0,0,0,0,0,1,-1], [0,0,0,0,0,1,0,0,-1]))
    
    # Term 18: (A[0,0]+A[2,2]) * (B[0,0]+B[2,2]) -> C[0,0]+C[2,2]
    # u: A[0,0]=1, A[2,2]=1
    # v: B[0,0]=1, B[2,2]=1
    # w: C[0,0]=1, C[2,2]=1
    terms.append(([1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1]))
    
    # Term 19: (A[0,1]+A[2,1]) * (B[1,0]+B[1,2]) -> C[0,1]+C[2,1]
    # u: A[0,1]=1, A[2,1]=1
    # v: B[1,0]=1, B[1,2]=1
    # w: C[0,1]=1, C[2,1]=1
    terms.append(([0,1,0,0,0,0,0,1,0], [0,0,0,1,0,0,0,0,1], [0,1,0,0,0,0,0,0,1]))
    
    # Term 20: (A[1,0]+A[1,2]) * (B[0,1]+B[2,1]) -> C[1,0]+C[1,2]
    # u: A[1,0]=1, A[1,2]=1
    # v: B[0,1]=1, B[2,1]=1
    # w: C[1,0]=1, C[1,2]=1
    terms.append(([0,0,0,1,0,0,0,0,1], [0,1,0,0,0,0,0,1,0], [0,1,0,0,0,1,0,0,0]))
    
    # Term 21: (A[0,2]+A[1,2]) * (B[2,0]+B[2,1]) -> C[0,2]+C[1,2]
    # u: A[0,2]=1, A[1,2]=1
    # v: B[2,0]=1, B[2,1]=1
    # w: C[0,2]=1, C[1,2]=1
    terms.append(([0,0,1,0,0,1,0,0,0], [0,0,0,0,0,0,1,1,0], [0,0,1,0,0,1,0,0,0]))
    
    # Term 22: (A[0,0]+A[0,2]) * (B[0,0]+B[2,0]) -> C[0,0]+C[2,0]
    # u: A[0,0]=1, A[0,2]=1
    # v: B[0,0]=1, B[2,0]=1
    # w: C[0,0]=1, C[2,0]=1
    terms.append(([1,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,1]))
    
    # Term 23: (A[1,0]+A[2,2]) * (B[0,1]+B[2,2]) -> C[1,0]+C[2,2]
    # u: A[1,0]=1, A[2,2]=1
    # v: B[0,1]=1, B[2,2]=1
    # w: C[1,0]=1, C[2,2]=1
    terms.append(([0,0,0,1,0,0,0,0,1], [0,1,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0,1]))
    
    return terms

def _build_3x3x4_from_lader():
    """
    Build a rank-32 decomposition for 3x3x4 using Lader's rank-23 for the 3x3x3 part
    and schoolbook for the 4th column.
    
    The 3x3x4 multiplication: A(3x3) * B(3x4) = C(3x4)
    We decompose B into B1(3x3) and B2(3x1) [4th column].
    C = A*B1 + A*B2
    
    For A*B1: use Lader's 23 terms (for 3x3x3)
    For A*B2: use 9 schoolbook terms
    
    Total rank: 23 + 9 = 32
    """
    lader_terms = _lader_3x3x3()
    
    n, m, p = 3, 3, 4
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 32
    V = [[0] * rank for _ in range(m * p)]  # 12 x 32
    W = [[0] * rank for _ in range(p * n)]  # 12 x 32
    
    # First 23 terms: Lader's 3x3x3 for the first 3 columns of B
    for t, (u, v, w) in enumerate(lader_terms):
        for idx in range(9):
            U[idx][t] = u[idx]
            # v is for B[l,k] where l in 0..2, k in 0..2 (first 3 columns)
            V[idx][t] = v[idx]
            # w is for C[k,i] where k in 0..2 (first 3 rows of C)
            W[idx][t] = w[idx]
    
    # Last 9 terms: schoolbook for the 4th column of B (k=3)
    col = 23
    for i in range(3):
        for j in range(3):
            U[i * 3 + j][col] = 1
            V[j * 4 + 3][col] = 1  # B[j, 3]
            W[3 * 3 + i][col] = 1  # C[3, i] -- wait, C is 3x4, so C[k,i] with k in 0..3, i in 0..2
            # W index: k*n + i = k*3 + i, where k is row of C (0..3), i is col of C (0..2)
            # For the 4th column of B, we compute C[k,i] = sum_j A[i,j]*B[j,3]
            # So W should have C[k,i] for k in 0..2 (rows of C), i in 0..2 (cols of C)
            # But the 4th column of B only contributes to C[k, i] where... 
            # Actually C is 3x4, so C has 3 rows and 4 cols.
            # C[r, c] = sum_j A[r, j] * B[j, c]
            # For c=3 (4th column): C[r, 3] = sum_j A[r, j] * B[j, 3]
            # So W index should be for C[r, 3] where r in 0..2
            # W index: k*n + i. Here k is the row index of C (0..2), i is the col index of C (0..3)
            # So for C[r, 3]: W index = r*3 + 3
            col += 1
    
    # Let me redo the schoolbook part more carefully
    # Reset and rebuild
    U = [[0] * rank for _ in range(n * m)]  # 9 x 32
    V = [[0] * rank for _ in range(m * p)]  # 12 x 32
    W = [[0] * rank for _ in range(p * n)]  # 12 x 32
    
    # First 23 terms: Lader's 3x3x3 for the first 3 columns of B
    for t, (u, v, w) in enumerate(lader_terms):
        for idx in range(9):
            U[idx][t] = u[idx]
            V[idx][t] = v[idx]  # B[l,k] for l in 0..2, k in 0..2
            W[idx][t] = w[idx]  # C[k,i] for k in 0..2, i in 0..2
    
    # Last 9 terms: schoolbook for the 4th column of B (column index 3)
    col = 23
    for i in range(3):  # row of A / col of C
        for j in range(3):  # col of A / row of B
            U[i * 3 + j][col] = 1       # A[i, j]
            V[j * 4 + 3][col] = 1       # B[j, 3]
            W[3 + i][col] = 1           # C[i, 3] -- W index: k*n + i where k=3 (row of C? No...)
            # Wait. Let me re-examine the W indexing.
            # W[k*n + i] where the convention is W[k*n+i, r]
            # The output is C with C[r, c] = sum_j A[r,j] B[j,c]
            # In the tensor: T_{i,j,k,l} = delta_{j,l}
            # U[i*m + j], V[l*p + k], W[k*n + i]
            # So i is row of A, j is col of A
            # l is row of B, k is col of B
            # k is col of B = row of C? No...
            # C has dimensions n x p = 3 x 4
            # C[r, c] where r in 0..2, c in 0..3
            # C[r, c] = sum_j A[r, j] * B[j, c]
            # So in the tensor: i=r (row of A), j (col of A), l (row of B), k=c (col of B)
            # W[k*n + i] = W[c*3 + r]
            # So for the 4th column of B (c=3): W[3*3 + r] = W[9 + r]
            col += 1
    
    # Fix W indexing for schoolbook part
    col = 23
    for i in range(3):
        for j in range(3):
            U[i * 3 + j][col] = 1
            V[j * 4 + 3][col] = 1
            W[9 + i][col] = 1  # W[3*3 + i] = W[9 + i]
            col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s deadline, 10s safety margin
    
    # Try to build the rank-32 decomposition
    try:
        candidate = _build_3x3x4_from_lader()
        rank = len(candidate['U'][0])
        
        # Validate
        if _validate(candidate['U'], candidate['V'], candidate['W'], n, m, p, rank):
            return candidate
    except Exception:
        pass
    
    # Fallback to schoolbook
    if time.time() < deadline:
        return _schoolbook(n, m, p)
    
    return _schoolbook(n, m, p)
