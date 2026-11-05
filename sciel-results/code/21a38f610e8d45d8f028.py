"""
Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.

Strategy:
  1. Use Strassen's rank-7 algorithm for 2x2x2 matrix multiplication.
  2. Use standard multiplication (rank 8) for 1x2 * 2x4 (which is equivalent to 2x2x2 with one row/col zeroed, but we use the Kronecker product structure).
  3. The rank of M_{2,4,8} can be bounded by the Kronecker product of M_{2,2,2} (rank 7) and M_{1,2,4} (rank 8), giving rank 56.
     More precisely, we can view 2x4 * 4x8 as two independent 2x4 * 4x4 blocks? No.
     Actually, the correct Kronecker product decomposition:
     M_{n*m, m*k, k*p} can be factored using M_{n,m,k} and M_{m,k,p}? No.
     
     The standard way: M_{n,m,p} where m = m1*m2, p = p1*p2, n = n1*n2.
     Here n=2, m=4, p=8. We can write n=2*1, m=2*2, p=4*2.
     Then M_{2,4,8} is isomorphic to the Kronecker product M_{2,2,4} ⊗ M_{1,2,2}? Not quite.
     
     Actually, the tensor for matrix multiplication is T_{n,m,p} with entries e_{i,j,k} where i in [n], j in [m], k in [p].
     We can decompose j = j1 + m1*j2, k = k1 + p1*k2, i = i1 + n1*i2.
     Then T = T_{n1,m1,p1} ⊗ T_{n2,m2,p2} if n=n1*n2, m=m1*m2, p=p1*p2.
     
     Here: n=2=2*1, m=4=2*2, p=8=4*2.
     So T_{2,4,8} = T_{2,2,4} ⊗ T_{1,2,2}.
     T_{2,2,4} can be further decomposed: n=2=2*1, m=2=2*1, p=4=2*2.
     So T_{2,2,4} = T_{2,2,2} ⊗ T_{1,2,2}.
     
     Thus T_{2,4,8} = T_{2,2,2} ⊗ T_{1,2,2} ⊗ T_{1,2,2}.
     
     Rank of T_{2,2,2} = 7 (Strassen).
     Rank of T_{1,2,2} = 4 (standard: 1x2 * 2x2 = 1x2, needs 4 mults).
     
     So total rank = 7 * 4 * 4 = 112. That's worse than schoolbook 64.
     
     Alternative: n=2=1*2, m=4=4*1, p=8=8*1.
     T_{2,4,8} = T_{1,4,8} ⊗ T_{2,1,1}. Rank = 32 * 2 = 64. Same as schoolbook.
     
     Better: n=2=2*1, m=4=2*2, p=8=2*4.
     T_{2,4,8} = T_{2,2,2} ⊗ T_{1,2,4}. Rank = 7 * 8 = 56.
     T_{1,2,4}: 1x2 * 2x4 = 1x4. Standard rank 8.
     
     So rank 56 < 64.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    U shape: (2*2, 7) = (4, 7), rows indexed by (i,j) where i in {0,1}, j in {0,1}
    V shape: (2*2, 7) = (4, 7), rows indexed by (j,k) where j in {0,1}, k in {0,1}
    W shape: (2*2, 7) = (4, 7), rows indexed by (k*n+i) = (k*2+i) where k in {0,1}, i in {0,1}
    
    Strassen's 7 multiplications:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Row indices: (i,j) -> i*2 + j for U and V
    # For W: (k,i) -> k*2 + i
    
    # M1: (A00+A11)*(B00+B11)
    # U: A00, A11 -> (0,0), (1,1) -> rows 0, 3
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    # V: B00, B11 -> (0,0), (1,1) -> rows 0, 3
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    # W: C00 += M1, C11 += M1 -> (0,0)=0, (1,1)=3
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2: (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1  # C11
    
    # M3: A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1  # -B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4: A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1  # -B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5: (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1  # C00
    W[1, 4] = 1  # C01
    
    # M6: (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # -A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7: (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # -A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    W[1, 6] = 1  # C01
    
    return U, V, W

def _standard_1x2x4():
    """
    Return U, V, W for 1x2 * 2x4 matrix multiplication with rank 8.
    U shape: (1*2, 8) = (2, 8), rows indexed by (i,j) where i=0, j in {0,1}
    V shape: (2*4, 8) = (8, 8), rows indexed by (j,k) where j in {0,1}, k in {0,1,2,3}
    W shape: (4*1, 8) = (4, 8), rows indexed by (k*n+i) = (k*1+0) = k where k in {0,1,2,3}
    
    C0 = A0*B00 + A1*B10
    C1 = A0*B01 + A1*B11
    C2 = A0*B02 + A1*B12
    C3 = A0*B03 + A1*B13
    
    8 multiplications:
    c=0: A0*B00
    c=1: A0*B01
    c=2: A0*B02
    c=3: A0*B03
    c=4: A1*B10
    c=5: A1*B11
    c=6: A1*B12
    c=7: A1*B13
    """
    U = np.zeros((2, 8))
    V = np.zeros((8, 8))
    W = np.zeros((4, 8))
    
    for c in range(8):
        if c < 4:
            # A0 * B0c
            U[0, c] = 1
            V[0 * 4 + c, c] = 1  # j=0, k=c
            W[c, c] = 1
        else:
            # A1 * B1(c-4)
            U[1, c] = 1
            V[1 * 4 + (c - 4), c] = 1  # j=1, k=c-4
            W[c - 4, c] = 1
    
    return U, V, W

def _kron_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Compute Kronecker product of two matrix multiplication decompositions.
    
    T_{n1,m1,p1} ⊗ T_{n2,m2,p2} = T_{n1*n2, m1*m2, p1*p2}
    
    Index mapping:
    i = i1 * n2 + i2, where i1 in [n1], i2 in [n2]
    j = j1 * m2 + j2, where j1 in [m1], j2 in [m2]
    k = k1 * p2 + k2, where k1 in [p1], k2 in [p2]
    
    U row: (i,j) -> i * m + j where m = m1*m2
    V row: (j,k) -> j * p + k where p = p1*p2
    W row: (k*n+i) where n = n1*n2
    
    c = c1 * rank2 + c2
    """
    n, m, p = n1 * n2, m1 * m2, p1 * p2
    rank1, rank2 = U1.shape[1], U2.shape[1]
    rank = rank1 * rank2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1 * rank2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for k1 in range(p1):
                        u1 = U1[i1 * m1 + j1, c1]
                        v1 = V1[j1 * p1 + k1, c1]
                        w1 = W1[k1 * n1 + i1, c1]
                        if u1 == 0 or v1 == 0 or w1 == 0:
                            continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                for k2 in range(p2):
                                    u2 = U2[i2 * m2 + j2, c2]
                                    v2 = V2[j2 * p2 + k2, c2]
                                    w2 = W2[k2 * n2 + i2, c2]
                                    if u2 == 0 or v2 == 0 or w2 == 0:
                                        continue
                                    
                                    i = i1 * n2 + i2
                                    j = j1 * m2 + j2
                                    k = k1 * p2 + k2
                                    
                                    val = u1 * v1 * w2  # Wait, need to be careful
                                    # Actually U, V, W are separate. 
                                    # U[i,j,c] = U1[i1,j1,c1] * U2[i2,j2,c2]
                                    # V[j,k,c] = V1[j1,k1,c1] * V2[j2,k2,c2]
                                    # W[k*n+i,c] = W1[k1*n1+i1,c1] * W2[k2*n2+i2,c2]
                                    
                                    U[i * m + j, c] = u1 * u2
                                    V[j * p + k, c] = v1 * v2
                                    W[k * n + i, c] = w1 * w2
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Compute rank-7 Strassen for 2x2x2
    U1, V1, W1 = _strassen_2x2x2()
    
    # Compute rank-8 standard for 1x2x4
    U2, V2, W2 = _standard_1x2x4()
    
    # Kronecker product: T_{2,2,2} ⊗ T_{1,2,4} = T_{2,4,8}
    # n1=2, m1=2, p1=2, n2=1, m2=2, p2=4
    U, V, W = _kron_product(U1, V1, W1, U2, V2, W2, 2, 2, 2, 1, 2, 4)
    
    # Convert to lists of ints/floats
    U_list = [[float(val) for val in row] for row in U]
    V_list = [[float(val) for val in row] for row in V]
    W_list = [[float(val) for val in row] for row in W]
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen, but return schoolbook as fallback
        n, m, p = DIMENSIONS
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_list, "V": V_list, "W": W_list}
