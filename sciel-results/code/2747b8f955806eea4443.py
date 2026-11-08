"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2 matrix multiplication.
    Each term is a tuple (u, v, w) where u, v, w are length-4 vectors.
    u corresponds to A[i,j] flattened (i*2+j).
    v corresponds to B[j,k] flattened (j*2+k).
    w corresponds to C[k,i] flattened (k*2+i).
    """
    terms = []
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # u: A00=1, A11=1 -> [1,0,0,1]
    # v: B00=1, B11=1 -> [1,0,0,1]
    # w: C00=1, C11=1 -> [1,0,0,1] (indices: 0->C00, 3->C11)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2: (A10+A11)B00 -> C10 += M2
    # u: A10=1, A11=1 -> [0,0,1,1]
    # v: B00=1 -> [1,0,0,0]
    # w: C10=1 -> index 1*2+0=2 -> [0,0,1,0]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # M3: A00(B01-B11) -> C01 += M3
    # u: A00=1 -> [1,0,0,0]
    # v: B01=1, B11=-1 -> [0,1,0,-1]
    # w: C01=1 -> index 0*2+1=1 -> [0,1,0,0]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # M4: A11(B10-B00) -> C00 += M4
    # u: A11=1 -> [0,0,0,1]
    # v: B10=1, B00=-1 -> [0,0,1,-1]
    # w: C00=1 -> index 0 -> [1,0,0,0]
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,0,0]))
    
    # M5: (A00+A01)B11 -> C11 += M5
    # u: A00=1, A01=1 -> [1,1,0,0]
    # v: B11=1 -> [0,0,0,1]
    # w: C11=1 -> index 3 -> [0,0,0,1]
    terms.append(([1,1,0,0], [0,0,0,1], [0,0,0,1]))
    
    # M6: (A10-A00)(B00+B01) -> C11 += M6
    # u: A10=1, A00=-1 -> [-1,0,1,0]
    # v: B00=1, B01=1 -> [1,1,0,0]
    # w: C11=1 -> index 3 -> [0,0,0,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7: (A01-A11)(B10+B11) -> C10 += M7
    # u: A01=1, A11=-1 -> [0,1,0,-1]
    # v: B10=1, B11=1 -> [0,0,1,1]
    # w: C10=1 -> index 2 -> [0,0,1,0]
    terms.append(([0,1,0,-1], [0,0,1,1], [0,0,1,0]))
    
    return terms

def _build_4x4x4_decomposition():
    """
    Builds the rank-49 decomposition for 4x4 x 4x4 matmul.
    Uses Kronecker product of two 2x2 Strassen decompositions.
    Returns U (16x49), V (16x49), W (16x49) as lists of lists.
    """
    s2 = _strassen_2x2_terms()
    R = 49
    U = [[0] * R for _ in range(16)]
    V = [[0] * R for _ in range(16)]
    W = [[0] * R for _ in range(16)]
    
    col = 0
    for (u1, v1, w1) in s2:
        for (u2, v2, w2) in s2:
            # u1, v1, w1 are length 4 (block level 2x2 indices)
            # u2, v2, w2 are length 4 (local level 2x2 indices)
            # Block indices I, J, K in {0,1}
            # Local indices i', j', k' in {0,1}
            # Global i = 2*I + i', j = 2*J + j', k = 2*K + k'
            # U index: i*4 + j = (2*I+i')*4 + (2*J+j') = 8*I + 4*i' + 2*J + j'
            # V index: j*4 + k = (2*J+j')*4 + (2*K+k') = 8*J + 4*j' + 2*K + k'
            # W index: k*4 + i = (2*K+k')*4 + (2*I+i') = 8*K + 4*k' + 2*I + i'
            
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        val1 = u1[I*2+J] * v1[J*2+K] * w1[K*2+I]
                        if val1 == 0: continue
                        for ip in range(2):
                            for jp in range(2):
                                for kp in range(2):
                                    val2 = u2[ip*2+jp] * v2[jp*2+kp] * w2[kp*2+ip]
                                    if val2 == 0: continue
                                    
                                    coeff = val1 * val2
                                    if coeff == 0: continue
                                    
                                    i = 2*I + ip
                                    j = 2*J + jp
                                    k = 2*K + kp
                                    
                                    u_idx = i*4 + j
                                    v_idx = j*4 + k
                                    w_idx = k*4 + i
                                    
                                    U[u_idx][col] += coeff
                                    V[v_idx][col] += coeff
                                    W[w_idx][col] += coeff
            col += 1
            
    # Convert to integers if possible
    for i in range(16):
        for c in range(49):
            if U[i][c] == int(U[i][c]): U[i][c] = int(U[i][c])
            if V[i][c] == int(V[i][c]): V[i][c] = int(V[i][c])
            if W[i][c] == int(W[i][c]): W[i][c] = int(W[i][c])
            
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """Construct rank-minimal decomposition for 4x4 x 4x8 matmul."""
    n, m, p = DIMENSIONS  # 4, 4, 8
    start_time = time.time()
    
    # Get 4x4x4 decomposition (rank 49)
    U4, V4, W4 = _build_4x4x4_decomposition()
    rank_4x4x4 = 49
    
    # Total rank for 4x4x8 is 2 * 49 = 98
    total_rank = 98
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * total_rank for _ in range(p * n)]  # 32 x 98
    
    # Fill first half (k=0..3)
    for c in range(rank_4x4x4):
        for i in range(n):
            for j in range(m):
                U[i*m + j][c] = U4[i*m + j][c]
        for j in range(m):
            for k in range(4):
                V[j*p + k][c] = V4[j*4 + k][c]
        for k in range(4):
            for i in range(n):
                W[k*n + i][c] = W4[k*n + i][c]
                
    # Fill second half (k=4..7)
    for c in range(rank_4x4x4):
        col = c + rank_4x4x4
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = U4[i*m + j][c]
        for j in range(m):
            for k in range(4):
                V[j*p + (k+4)][col] = V4[j*4 + k][c]
        for k in range(4):
            for i in range(n):
                W[(k+4)*n + i][col] = W4[k*n + i][c]
                
    # Verify
    if not _verify_decomp(U, V, W, n, m, p):
        # Fallback to schoolbook (rank 128)
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
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
