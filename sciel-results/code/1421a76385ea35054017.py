import numpy as np

DIMENSIONS = [4, 4, 7]

def _kron_strassen_2x2():
    """Return the 7 Kronecker-product Strassen terms for 4x4x4.
    Each term is (u, v, w) where u has 16 entries, v has 16 entries, w has 16 entries.
    The 2x2x2 Strassen decomposition has 7 terms. We Kronecker with 2x2 identity blocks.
    """
    # 2x2x2 Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each 2x2x2 term has vectors of length 4 for U, V, W.
    # For 4x4x4 via Kronecker with 2x2, each 2x2 block gets the same 2x2x2 structure.
    # So for each of the 7 terms, we repeat the pattern in each of the 4 2x2 blocks.
    
    # 2x2x2 terms (u, v, w) each of length 4:
    # Index convention: u[i*m+j] for A[i][j], m=2, so u[i*2+j]
    # v[j*p+k] for B[j][k], p=2, so v[j*2+k]
    # w[k*n+i] for C[k][i], n=2, so w[k*2+i]
    
    terms_2x2 = [
        # M1: (A11+A22)(B11+B22)
        # u: A11=1, A22=1 -> u[0]=1, u[3]=1
        # v: B11=1, B22=1 -> v[0]=1, v[3]=1
        # w: C11 += M1, C22 += M1 -> w[0]=1, w[3]=1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A21+A22)B11
        # u: A21=1, A22=1 -> u[2]=1, u[3]=1
        # v: B11=1 -> v[0]=1
        # w: C21 += M2 -> w[2]=1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3: A11(B21-B22)
        # u: A11=1 -> u[0]=1
        # v: B21=1, B22=-1 -> v[2]=1, v[3]=-1
        # w: C12 += M3 -> w[1]=1
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # M4: A22(B12-B11)
        # u: A22=1 -> u[3]=1
        # v: B12=1, B11=-1 -> v[1]=1, v[0]=-1
        # w: C22 += M4 -> w[3]=1
        ([0, 0, 0, 1], [-1, 1, 0, 0], [0, 0, 0, 1]),
        # M5: (A11+A12)B22
        # u: A11=1, A12=1 -> u[0]=1, u[1]=1
        # v: B22=1 -> v[3]=1
        # w: C12 += M5 -> w[1]=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        # M6: (A21-A11)(B11+B12)
        # u: A21=1, A11=-1 -> u[2]=1, u[0]=-1
        # v: B11=1, B12=1 -> v[0]=1, v[1]=1
        # w: C21 += M6 -> w[2]=1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),
        # M7: (A12-A22)(B21+B22)
        # u: A12=1, A22=-1 -> u[1]=1, u[3]=-1
        # v: B21=1, B22=1 -> v[2]=1, v[3]=1
        # w: C12 += M7 -> w[1]=1
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),
    ]
    
    # Now Kronecker with 2x2 to get 4x4x4.
    # For each 2x2x2 term, we create 4 copies (one for each 2x2 block position).
    # Actually, the Kronecker product structure means:
    # If we have a 2x2x2 decomposition, the 4x4x4 decomposition via block Kronecker
    # with 2x2 identity means each 2x2 block of A, B, C uses the same 2x2x2 terms.
    # So for each of the 7 terms, we repeat the u, v, w vectors in each of the 4 quadrants.
    
    # For 4x4x4: n=4, m=4, p=4
    # u has 16 entries, v has 16 entries, w has 16 entries
    # Block (r,c) where r,c in {0,1} for 2x2 blocks:
    # u index: (r*2 + i) * 4 + (c*2 + j) for i,j in {0,1}
    # v index: (c*2 + j) * 4 + (s*2 + k) for s in {0,1}
    # w index: (s*2 + k) * 4 + (r*2 + i) for r in {0,1}
    
    terms_4x4 = []
    for u2, v2, w2 in terms_2x2:
        u4 = [0] * 16
        v4 = [0] * 16
        w4 = [0] * 16
        for r in range(2):
            for c in range(2):
                for s in range(2):
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                # u index: (r*2+i)*4 + (c*2+j)
                                ui = (r*2 + i) * 4 + (c*2 + j)
                                # v index: (c*2+j)*4 + (s*2+k)
                                vi = (c*2 + j) * 4 + (s*2 + k)
                                # w index: (s*2+k)*4 + (r*2+i)
                                wi = (s*2 + k) * 4 + (r*2 + i)
                                # The 2x2x2 term contributes u2[i*2+j] * v2[j*2+k] * w2[k*2+i]
                                # But in the tensor decomposition, each term is a rank-1 tensor.
                                # The Kronecker product of rank-1 tensors is rank-1.
                                # So for each (r,c,s) block, we add the contribution.
                                u4[ui] += u2[i * 2 + j]
                                v4[vi] += v2[j * 2 + k]
                                w4[wi] += w2[k * 2 + i]
        terms_4x4.append((u4, v4, w4))
    
    return terms_4x4

def _schoolbook_4x4x3():
    """Return schoolbook decomposition for 4x4x3.
    n=4, m=4, p=3. Rank = 4*4*3 = 48.
    u has 16 entries, v has 12 entries, w has 12 entries.
    """
    n, m, p = 4, 4, 3
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                terms.append((u, v, w))
    return terms

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Use Strassen for first 4 columns (4x4x4, rank 49)
    # Use schoolbook for last 3 columns (4x4x3, rank 48)
    # Total rank = 97
    
    strassen_terms = _kron_strassen_2x2()
    schoolbook_terms = _schoolbook_4x4x3()
    
    rank = len(strassen_terms) + len(schoolbook_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill Strassen part
    for col, (u, v, w) in enumerate(strassen_terms):
        for i in range(n * m):
            U[i][col] = u[i]
        for i in range(16):  # first 4 columns of B (4x4 part)
            V[i][col] = v[i]
        for i in range(16):  # first 4 columns of C (4x4 part)
            W[i][col] = w[i]
    
    # Fill schoolbook part
    for col_idx, (u, v, w) in enumerate(schoolbook_terms):
        col = len(strassen_terms) + col_idx
        for i in range(n * m):
            U[i][col] = u[i]
        for i in range(12):  # last 3 columns of B (4x3 part), indices 16..27
            V[16 + i][col] = v[i]
        for i in range(12):  # last 3 columns of C (4x3 part), indices 16..27
            W[16 + i][col] = w[i]
    
    return {"U": U, "V": V, "W": W}
