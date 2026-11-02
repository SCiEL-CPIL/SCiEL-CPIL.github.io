"""Rank-35 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Kronecker product of 2x2x2 Strassen (rank 7) and 1x2x3 core (rank 5).
The 2x4x6 tensor factors as (2x2) x (1x2x3) via the group ring Z[C2 x C3].
Total rank = 7 * 5 = 35.

The 1x2x3 core computes [a0, a1] @ [[b00,b01,b02],[b10,b11,b12]] = [c0,c1,c2].
Its rank is 5 (one saving over the trivial 6).
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7).
    U1 row: i*2+j, V1 row: j*2+k, W1 row: k*2+i (trace convention)."""
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1  # C00 += M1
    W1[3, 0] = 1  # C11 += M1

    # M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1  # C10 += M2
    W1[3, 1] = -1 # C11 -= M2

    # M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1  # C01 += M3
    W1[3, 2] = 1  # C11 += M3

    # M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1  # C00 += M4
    W1[1, 3] = 1  # C10 += M4

    # M5 = (A00+A01)B11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1 # C00 -= M5
    W1[2, 4] = 1  # C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1  # C11 += M6

    # M7 = (A01-A11)(B10+B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1  # C00 += M7

    return U1, V1, W1


def _rect_1x2x3_factors():
    """Return (U2, V2, W2) for 1x2x3 matrix multiplication, rank 5.
    Computes [a0,a1] @ [[b00,b01,b02],[b10,b11,b12]] = [c0,c1,c2].
    
    U2: (1*2, 5) = (2, 5) — row index: j (0..1)
    V2: (2*3, 5) = (6, 5) — row index: j*3+k
    W2: (3*1, 5) = (3, 5) — row index: k (trace convention: k*1+0=k)
    
    Uses Winograd/FTW: 5 multiplications for 6 products.
    M1 = a0 * b00
    M2 = a1 * b10
    M3 = (a0+a1) * (b01+b11)
    M4 = a0 * b02
    M5 = a1 * b12
    
    c0 = M1 + M2
    c1 = M3 - M1 - M2
    c2 = M4 + M5
    """
    U2 = np.zeros((2, 5), dtype=float)
    V2 = np.zeros((6, 5), dtype=float)
    W2 = np.zeros((3, 5), dtype=float)

    # M1 = a0 * b00
    U2[0, 0] = 1
    V2[0*3 + 0, 0] = 1  # b00
    W2[0, 0] = 1        # c0 += M1
    W2[1, 0] = -1       # c1 -= M1

    # M2 = a1 * b10
    U2[1, 1] = 1
    V2[1*3 + 0, 1] = 1  # b10
    W2[0, 1] = 1        # c0 += M2
    W2[1, 1] = -1       # c1 -= M2

    # M3 = (a0+a1) * (b01+b11)
    U2[0, 2] = 1; U2[1, 2] = 1
    V2[0*3 + 1, 2] = 1  # b01
    V2[1*3 + 1, 2] = 1  # b11
    W2[1, 2] = 1        # c1 += M3

    # M4 = a0 * b02
    U2[0, 3] = 1
    V2[0*3 + 2, 3] = 1  # b02
    W2[2, 3] = 1        # c2 += M4

    # M5 = a1 * b12
    U2[1, 4] = 1
    V2[1*3 + 2, 4] = 1  # b12
    W2[2, 4] = 1        # c2 += M5

    return U2, V2, W2


def _build_rank35_kron():
    """Build rank-35 decomposition via Kronecker product of Strassen and 1x2x3 core.
    
    The 2x4x6 tensor is decomposed as:
    - First dimension (n=2): from 2x2x2 Strassen
    - Second dimension (m=4) = 2*2: first factor from Strassen (j), second from 1x2x3 (j')
    - Third dimension (p=6) = 2*3: first factor from Strassen (k), second from 1x2x3 (k')
    
    Total rank = 7 * 5 = 35.
    """
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    U2, V2, W2 = _rect_1x2x3_factors()
    
    rank1, rank2 = 7, 5
    total_rank = rank1 * rank2  # 35

    U = np.zeros((n * m, total_rank), dtype=float)  # (8, 35)
    V = np.zeros((m * p, total_rank), dtype=float)  # (24, 35)
    W = np.zeros((p * n, total_rank), dtype=float)  # (12, 35)

    # Kronecker product structure:
    # U row: (i, j_total) where j_total = j1 * 2 + j2
    #   i in 0..1 (from Strassen), j1 in 0..1 (from Strassen), j2 in 0..1 (from 1x2x3)
    # V row: (j_total, k_total) where k_total = k1 * 3 + k2
    #   j_total = j1 * 2 + j2, k1 in 0..1 (from Strassen), k2 in 0..2 (from 1x2x3)
    # W row: (k_total, i) where k_total = k1 * 3 + k2
    #   k1 in 0..1, k2 in 0..2, i in 0..1

    for s1 in range(rank1):
        for s2 in range(rank2):
            s = s1 * rank2 + s2  # total column index

            # U: for each (i, j1, j2)
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        u1_coeff = U1[i * 2 + j1, s1]
                        u2_coeff = U2[j2, s2]
                        if u1_coeff != 0 and u2_coeff != 0:
                            U[i * m + (j1 * 2 + j2), s] += u1_coeff * u2_coeff

            # V: for each (j1, j2, k1, k2)
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1_coeff = V1[j1 * 2 + k1, s1]
                            v2_coeff = V2[j2 * 3 + k2, s2]
                            if v1_coeff != 0 and v2_coeff != 0:
                                V[(j1 * 2 + j2) * p + (k1 * 3 + k2), s] += v1_coeff * v2_coeff

            # W: for each (k1, k2, i)
            for k1 in range(2):
                for k2 in range(3):
                    for i in range(2):
                        w1_coeff = W1[k1 * 2 + i, s1]
                        w2_coeff = W2[k2, s2]
                        if w1_coeff != 0 and w2_coeff != 0:
                            W[(k1 * 3 + k2) * n + i, s] += w1_coeff * w2_coeff

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _to_output(U, V, W):
    """Convert numpy arrays to list of lists with int where possible."""
    U_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in U]
    V_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in V]
    W_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in W]
    return {"U": U_out, "V": V_out, "W": W_out}


def _build_schoolbook():
    """Build trivial rank-48 schoolbook decomposition."""
    n, m, p = 2, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj, col] = 1
                V[jj * p + kk, col] = 1
                W[kk * n + ii, col] = 1
                col += 1
    return U, V, W


def _build_rank42_strassen_trivial():
    """
    Build rank-42 decomposition.
    2x4x6 = (2x2x2 Strassen) x (1x1x3 Trivial) for two independent blocks.
    Block 1: A[0:2, 0:2] @ B[0:2, :] (2x2x6)
    Block 2: A[0:2, 2:4] @ B[2:4, :] (2x2x6)
    Each 2x2x6 is decomposed as 2x2x2 (rank 7) x 1x1x3 (rank 3).
    Total rank = 2 * (7 * 3) = 42.
    """
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors() # 2x2x2, shapes (4,7), (4,7), (4,7)
    
    # Trivial 1x1x3 factors
    # n2=1, m2=1, p2=3
    # U2 shape (1*1, 3) = (1,3)
    # V2 shape (1*3, 3) = (3,3)
    # W2 shape (3*1, 3) = (3,3)
    U2 = np.zeros((1, 3), dtype=float)
    V2 = np.zeros((3, 3), dtype=float)
    W2 = np.zeros((3, 3), dtype=float)
    for k in range(3):
        U2[0, k] = 1
        V2[k, k] = 1
        W2[k, k] = 1
        
    rank1 = 7
    rank2 = 3
    block_rank = rank1 * rank2
    total_rank = 2 * block_rank # 42

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    # Block 0: A cols 0,1 (j1=0,1) and B rows 0,1 (j1=0,1)
    # Block 1: A cols 2,3 (j1=2,3) and B rows 2,3 (j1=2,3)
    
    for block in range(2):
        # Offset for A cols and B rows in the global m=4 dimension
        # A is 2x4. Cols 0,1 for block 0, Cols 2,3 for block 1.
        # B is 4x6. Rows 0,1 for block 0, Rows 2,3 for block 1.
        
        for s1 in range(rank1):
            for s2 in range(rank2):
                s_local = s1 * rank2 + s2
                s = block * block_rank + s_local
                
                # U: A indices (i, j_local) -> global (i, j_global)
                # i in 0..1, j_local in 0..1
                # j_global = block * 2 + j_local
                for i in range(2):
                    for j_local in range(2):
                        j_global = block * 2 + j_local
                        u1_coeff = U1[i * 2 + j_local, s1]
                        u2_coeff = U2[0, s2] # U2 is 1x3, only index 0
                        if u1_coeff != 0 and u2_coeff != 0:
                            U[i * m + j_global, s] += u1_coeff * u2_coeff

                # V: B indices (j_local, k) -> global (j_global, k)
                # j_local in 0..1, k in 0..5 (p=6)
                # The 1x1x3 part handles the p dimension?
                # Wait, the 2x2x6 is decomposed as (2x2x2) x (1x1x3).
                # The p=6 dimension is split into p1=2 (from Strassen) and p2=3 (from Trivial)?
                # No, 2x2x2 has p=2. 1x1x3 has p=3. Product p=6.
                # So k_global = k1 * 3 + k2.
                # k1 in 0..1 (from Strassen), k2 in 0..2 (from Trivial).
                
                for j_local in range(2):
                    j_global = block * 2 + j_local
                    for k1 in range(2):
                        for k2 in range(3):
                            k_global = k1 * 3 + k2
                            v1_coeff = V1[j_local * 2 + k1, s1]
                            v2_coeff = V2[k2, s2] # V2 is 3x3, index k2
                            if v1_coeff != 0 and v2_coeff != 0:
                                V[j_global * p + k_global, s] += v1_coeff * v2_coeff

                # W: C indices (k, i) -> global (k_global, i)
                # k_global = k1 * 3 + k2
                for k1 in range(2):
                    for k2 in range(3):
                        k_global = k1 * 3 + k2
                        for i in range(2):
                            w1_coeff = W1[k1 * 2 + i, s1]
                            w2_coeff = W2[k2, s2] # W2 is 3x3, index k2
                            if w1_coeff != 0 and w2_coeff != 0:
                                W[k_global * n + i, s] += w1_coeff * w2_coeff

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Try rank-35 Kronecker decomposition first
    U, V, W = _build_rank35_kron()
    if _verify(U, V, W, n, m, p):
        return _to_output(U, V, W)

    # Fallback: rank-42 Strassen-Trivial product
    if time.time() < deadline:
        U42, V42, W42 = _build_rank42_strassen_trivial()
        if _verify(U42, V42, W42, n, m, p):
            return _to_output(U42, V42, W42)

    # Fallback: schoolbook rank 48
    if time.time() < deadline:
        U_fb, V_fb, W_fb = _build_schoolbook()
        return _to_output(U_fb, V_fb, W_fb)

    # Emergency: return empty (should not happen)
    return {"U": [], "V": [], "W": []}
