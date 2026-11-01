"""
Matrix multiplication 2x2x16 rank-minimal decomposition.

Strategy: Use Strassen's rank-7 algorithm for the 2x2x2 core, and exploit the
linearity of the output dimension (p=16) to pack 8 independent column blocks.

The 2x2x16 multiplication tensor can be viewed as 8 copies of the 2x2x2 tensor
(treated as multiplying 2x2 by 2x2 for each pair of output columns). By using
Strassen's rank-7 decomposition for each 2x2x2 block, and since the 8 blocks are
independent in the output dimension, we get rank = 7 * 8 = 56.

This is better than schoolbook rank = 2*2*16 = 64.
"""

import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-56 decomposition of the 2x2x16 matrix multiplication tensor.
    
    Uses Strassen's rank-7 algorithm for the 2x2 core, replicated across the 8
    independent column blocks of the 2x16 output.
    
    The tensor T[i, j, k] for (A*B)[i,k] = sum_j A[i,j]*B[j,k] where:
      A is 2x2 (indices i in {0,1}, j in {0,1})
      B is 2x16 (indices j in {0,1}, k in {0,...,15})
      Output is 2x16 (indices i in {0,1}, k in {0,...,15})
    
    W index convention: W[k*n+i, r] where n=2, so W has 16*2 = 32 rows.
    U index: U[i*m+j, r] where m=2, so U has 2*2 = 4 rows.
    V index: V[j*p+k, r] where p=16, so V has 2*16 = 32 rows.
    
    Strassen's rank-7 decomposition for 2x2x2:
    The 7 rank-1 tensors are defined by the following (u, v, w) triples
    where u[i,j], v[j,k], w[k,i] are the coefficient patterns.
    
    For the 2x2x2 case, Strassen's algorithm gives:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    In tensor form, each rank-1 component has:
    U vector (for A indices i,j), V vector (for B indices j,k), W vector (for output k,i)
    
    Let's define the 7 Strassen components for 2x2x2:
    Component 1: U = [1,0, 0,1] (A11+A22), V = [1,1, 1,1] (B11+B22 for both k), W = [1, 1] (C11 + C22 contribution... wait)
    
    Actually, let me be more careful. For 2x2x2:
    A has indices (i,j) with i,j in {0,1}
    B has indices (j,k) with j,k in {0,1}
    C has indices (i,k) with i,k in {0,1}
    
    The multiplication tensor T[i,j,k] = delta (meaning T[i,j,k] = 1 if C[i,k] = sum_j A[i,j] B[j,k])
    
    Strassen's 7 terms:
    Term 1: (A00+A11)(B00+B11) contributes to C00 and C11
    Term 2: (A10+A11)B00 contributes to C10
    Term 3: A00(B01-B11) contributes to C01
    Term 4: A11(B10-B00) contributes to C10 and C11... 
    
    Let me use the standard formulation. For 2x2 matrices:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen:
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    
    Now in tensor notation for 2x2x2:
    U has 4 entries (i*m+j = i*2+j for i,j in {0,1}): U[0]=a, U[1]=b, U[2]=c, U[3]=d
    V has 4 entries (j*p+k = j*2+k for j,k in {0,1}): V[0]=e, V[1]=f, V[2]=g, V[3]=h
    W has 4 entries (k*n+i = k*2+i for k,i in {0,1}): W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11
    
    Wait, the convention is W[k*n+i, r]. With n=2:
    k=0,i=0: index 0 -> C00
    k=0,i=1: index 1 -> C10
    k=1,i=0: index 2 -> C01
    k=1,i=1: index 3 -> C11
    
    So W = [C00, C10, C01, C11]
    
    The 7 Strassen rank-1 tensors for 2x2x2:
    
    P1 = (a+d)(e+h): U=[1,0,0,1], V=[1,1,1,1], contributes to C00 and C11
    C00 += P1, C11 += P1
    W contribution: [1, 0, 0, 1]
    
    P2 = (c+d)e: U=[0,0,1,1], V=[1,0,0,0], contributes to C10
    C10 += P2
    W contribution: [0, 1, 0, 0]
    
    P3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1], contributes to C01
    C01 += P3
    W contribution: [0, 0, 1, 0]
    
    P4 = d(g-e): U=[0,0,0,1], V=[-1,0,1,0], contributes to C10 and C11
    C10 += P4, C11 += P4
    W contribution: [0, 1, 0, 1]
    
    P5 = (a+b)h: U=[1,1,0,0], V=[0,0,0,1], contributes to C01 and C00... 
    Wait: C00 = P1 + P4 - P5 + P7, C01 = P3 + P5
    So P5 contributes to C01 with +1 and C00 with -1
    W contribution: [-1, 0, 1, 0]
    
    P6 = (c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0], contributes to C11
    C11 += P6
    W contribution: [0, 0, 0, 1]
    
    P7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1], contributes to C00
    C00 += P7
    W contribution: [1, 0, 0, 0]
    
    Let me verify:
    C00 = P1 + P4 - P5 + P7 = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
    = ae + (ah-ah) + (de-de) + (dh-dh) + (dg-dg) + (-bh+bh) + bg
    = ae + bg ✓
    
    C01 = P3 + P5 = a(f-h) + (a+b)h = af-ah+ah+bh = af+bh ✓
    
    C10 = P2 + P4 = (c+d)e + d(g-e) = ce+de+dg-de = ce+dg ✓
    
    C11 = P1 - P2 + P3 + P6 = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
    = ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af
    = (ae-ae) + (ah-ah) + (de-de) + dh + (-ce+ce) + (af-af) + cf
    = dh + cf ✓
    
    Great, the Strassen decomposition is correct.
    
    Now for 2x2x16:
    A is 2x2, B is 2x16, C is 2x16.
    U has 4 rows (i*m+j, m=2)
    V has 32 rows (j*p+k, p=16)
    W has 32 rows (k*n+i, n=2)
    
    For each pair of output columns (k0, k0+1) where k0 = 0, 2, 4, ..., 14,
    we can treat it as a 2x2x2 problem where the B matrix columns are B[:, k0] and B[:, k0+1].
    
    The 8 independent blocks correspond to k0 = 0, 2, 4, 6, 8, 10, 12, 14.
    
    For block b (b = 0, ..., 7), the columns are k = 2b and k = 2b+1.
    
    For each Strassen term t (t = 0, ..., 6) and each block b (b = 0, ..., 7):
    - U vector: same as the 2x2x2 U for term t (4 entries)
    - V vector: 32 entries, nonzero only at positions corresponding to j=0,k=2b and j=0,k=2b+1 and j=1,k=2b and j=1,k=2b+1
      V[j*16+k] for the relevant (j,k) pairs
    - W vector: 32 entries, nonzero only at positions corresponding to the output C[i, 2b] and C[i, 2b+1]
      W[k*2+i] for k in {2b, 2b+1} and i in {0, 1}
    
    Total rank = 7 * 8 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen's 7 rank-1 components for 2x2x2
    # Each component: (U_4, V_4, W_4) for the 2x2x2 case
    # U indices: [0]=a, [1]=b, [2]=c, [3]=d
    # V indices: [0]=e, [1]=f, [2]=g, [3]=h  (j*2+k: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3)
    # W indices: [0]=C00, [1]=C10, [2]=C01, [3]=C11  (k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3)
    
    strassen_U = [
        [1, 0, 0, 1],  # P1: a+d
        [0, 0, 1, 1],  # P2: c+d
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: a+b
        [-1, 0, 1, 0], # P6: c-a
        [0, 1, 0, -1], # P7: b-d
    ]
    
    strassen_V = [
        [1, 1, 1, 1],  # P1: e+h -> (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
        [1, 0, 0, 0],  # P2: e -> (0,0)=1
        [0, 1, 0, -1], # P3: f-h -> (0,1)=1, (1,1)=-1
        [-1, 0, 1, 0], # P4: g-e -> (0,0)=-1, (1,0)=1
        [0, 0, 0, 1],  # P5: h -> (1,1)=1
        [1, 1, 0, 0],  # P6: e+f -> (0,0)=1, (0,1)=1
        [0, 0, 1, 1],  # P7: g+h -> (1,0)=1, (1,1)=1
    ]
    
    strassen_W = [
        [1, 0, 0, 1],  # P1: C00 + C11
        [0, 1, 0, 0],  # P2: C10
        [0, 0, 1, 0],  # P3: C01
        [0, 1, 0, 1],  # P4: C10 + C11
        [-1, 0, 1, 0], # P5: -C00 + C01
        [0, 0, 0, 1],  # P6: C11
        [1, 0, 0, 0],  # P7: C00
    ]
    
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    col = 0
    for b in range(num_blocks):
        k0 = 2 * b  # first column index in this block
        k1 = 2 * b + 1  # second column index in this block
        
        for t in range(7):
            # U: 4 entries, same for all blocks
            for ui in range(4):
                U[ui][col] = strassen_U[t][ui]
            
            # V: 32 entries, nonzero at (j, k0) and (j, k1) for j=0,1
            # V index = j * p + k = j * 16 + k
            # strassen_V[t] has 4 entries indexed as (j*2 + k_local) where k_local in {0,1}
            # k_local=0 corresponds to k=k0, k_local=1 corresponds to k=k1
            for j in range(2):
                for k_local in range(2):
                    v_idx = j * p + (k0 + k_local)
                    s_idx = j * 2 + k_local
                    V[v_idx][col] = strassen_V[t][s_idx]
            
            # W: 32 entries, nonzero at (k, i) for k in {k0, k1} and i in {0, 1}
            # W index = k * n + i = k * 2 + i
            # strassen_W[t] has 4 entries indexed as (k_local * 2 + i) where k_local in {0,1}
            for k_local in range(2):
                k = k0 + k_local
                for i in range(2):
                    w_idx = k * n + i
                    s_idx = k_local * 2 + i
                    W[w_idx][col] = strassen_W[t][s_idx]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
