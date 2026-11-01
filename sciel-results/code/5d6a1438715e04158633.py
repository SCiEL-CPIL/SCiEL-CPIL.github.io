"""Rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time

def _build_block_decomp():
    """
    Build a block decomposition of the 2x3 * 3x10 multiplication tensor.
    
    Strategy: Split the 10 columns of the second matrix into 5 blocks of 2 columns.
    For each block, we multiply a 2x3 matrix by a 3x2 matrix.
    A 2x3 * 3x2 multiplication can be done with rank 4 using a clever 
    decomposition (exploiting that we only need 2 output columns).
    
    For a 2x3 * 3x2 multiplication (producing a 2x2 result):
    Let A be 2x3, B be 3x2. Result C = A*B is 2x2.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    Schoolbook rank for 2x3*3x2 is 4 (one per output entry).
    But we can do better: use 4 terms where each term is a product of
    linear combinations. Actually, let me think more carefully.
    
    The tensor for 2x3 * 3x2 has 4 entries. The minimum rank is at least 4
    (each output entry needs its own term unless we can share).
    
    Actually, we can use a rank-4 decomposition trivially. But can we do rank 3?
    Unlikely for 4 independent bilinear forms.
    
    So 5 blocks * 4 = rank 20. That's a big improvement from 60!
    
    Wait, let me reconsider. The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,9}.
    We need sum_c U[i*3+j, c] * V[j*10+k, c] * W[k*2+i, c] = 1 for all (i,j,k).
    
    With block structure: for k in block b (k = 2*b + kk, kk in {0,1}):
    W[k*2+i, c] = W[(2*b+kk)*2+i, c] = W[4*b+2*kk+i, c]
    
    Let me use a cleaner approach. For each of the 5 output column pairs (block b),
    I'll use 4 rank terms (one per output entry in that 2x2 block).
    
    For block b, the 4 terms correspond to:
    (i=0,kk=0), (i=0,kk=1), (i=1,kk=0), (i=1,kk=1)
    
    For term (i,kk) in block b, column c = 4*b + 2*i + kk:
    U[i*3+j, c] = 1 for all j (this doesn't work directly...)
    
    Let me think differently. The standard schoolbook uses 1 term per (i,j,k) triple.
    For block b, we have 2*3*2 = 12 triples, but we want to reduce this.
    
    Actually, the key insight from the inspiration: use Strassen-like ideas.
    For 2x3 * 3x2, we can potentially use rank 4 (one per output).
    
    Let me just use rank 4 per block with a proper construction:
    
    For block b (columns 2b, 2b+1 of B):
    Output C[i, 2b] = sum_j A[i,j] * B[j, 2b]
    Output C[i, 2b+1] = sum_j A[i,j] * B[j, 2b+1]
    
    Rank-4 decomposition for one block:
    Term 1 (i=0, kk=0): 
      U[0*3+j, c] = 1 for j=0,1,2
      V[j*10+2b, c] = 1 for j=0,1,2
      W[(2b)*2+0, c] = 1, others 0
    This gives: sum_j 1*1*1 = 3... no, that's not right.
    
    I need: sum_j U[0*3+j, c] * V[j*10+2b, c] * W[2b*2+0, c] = 1
    And: sum_j U[1*3+j, c'] * V[j*10+2b, c'] * W[2b*2+0, c'] = 0 (for the other i)
    
    This is getting complex. Let me just use the schoolbook but grouped.
    
    Actually, the simplest correct approach: use 4 terms per block where each term
    isolates one (i, kk) pair but sums over j.
    
    For term corresponding to (i, kk) in block b, column c:
    U[i*3+j, c] = 1 for all j in {0,1,2}
    V[j*10+(2b+kk), c] = 1 for all j in {0,1,2}  
    W[(2b+kk)*2+i, c] = 1
    W[(2b+kk)*2+(1-i), c] = 0
    
    Then sum_j U[i*3+j,c]*V[j*10+2b+kk,c]*W[(2b+kk)*2+i,c] = sum_j 1*1*1 = 3
    
    That gives 3, not 1. I need to scale by 1/3.
    
    So U[i*3+j, c] = 1/3, V[j*10+2b+kk, c] = 1, W[(2b+kk)*2+i, c] = 1
    Then sum = sum_j (1/3)*1*1 = 1. 
    
    But we need W[(2b+kk)*2+(1-i), c] = 0, which is fine.
    
    Wait, but for a different i', the term contributes:
    sum_j U[i'*3+j, c] * V[j*10+2b+kk, c] * W[(2b+kk)*2+i', c]
    
    If i' != i, U[i'*3+j, c] = 0 (since U is nonzero only for row i*3+j), so contribution is 0. Good.
    
    If i' = i but different kk' or block b', then W[(2b'+kk')*2+i, c] = 0 unless (b',kk')=(b,kk). Good.
    
    So this works! Each block uses 4 terms, total rank = 5*4 = 20.
    """
    n, m, p = 2, 3, 10
    rank = 20
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 20
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 20
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 20
    
    for b in range(5):  # 5 blocks of 2 columns
        for i in range(2):
            for kk in range(2):
                c = 4 * b + 2 * i + kk
                k = 2 * b + kk
                # U: nonzero for rows i*3+j, j in {0,1,2}
                for j in range(m):
                    U[i * m + j][c] = Fraction(1, 3)
                # V: nonzero for rows j*p+k, j in {0,1,2}
                for j in range(m):
                    V[j * p + k][c] = 1
                # W: nonzero only for row k*n+i
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    from fractions import Fraction
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 140
    
    # Try the block decomposition (rank 20)
    best = _build_block_decomp()
    best_rank = 20
    
    # Verify with exact arithmetic
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        best = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    return best
