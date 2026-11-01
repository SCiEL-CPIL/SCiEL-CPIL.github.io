import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, tol=1e-8):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    recon = np.einsum('ijr,jkr,krir->ijk', U4, V4, W4)
    target = np.ones((n, m, p))
    return np.max(np.abs(recon - target)) < tol


def _schoolbook_64():
    """Return the rank-64 schoolbook decomposition."""
    n, m, p = 4, 4, 4
    rank = 64
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _construct_rank49_strassen():
    """
    Construct the rank-49 Strassen decomposition for 4x4 matrix multiplication.
    
    The 4x4 multiplication tensor is the Kronecker square of the 2x2 multiplication tensor.
    The 2x2 tensor has rank 7 (Strassen), so the 4x4 tensor has rank at most 7*7 = 49.
    
    We build the factor matrices directly using the Kronecker product structure.
    
    For the 2x2 case:
    - U_2: shape (4, 7) - entries U_2[i*2+j, r] for i,j in {0,1}, r in {0..6}
    - V_2: shape (4, 7) - entries V_2[j*2+k, r] for j,k in {0,1}, r in {0..6}
    - W_2: shape (4, 7) - entries W_2[k*2+i, r] for k,i in {0,1}, r in {0..6}
    
    Strassen's 7 multiplication terms for 2x2:
    Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    m1 = (a11+a22)*(b11+b22)
    m2 = (a21+a22)*b11
    m3 = a11*(b12-b22)
    m4 = a22*(b21-b11)
    m5 = (a11+a12)*b22
    m6 = (a21-a11)*(b11+b12)
    m7 = (a12-a22)*(b21+b22)
    
    C = [[m1+m4-m5+m7, m3+m5],
         [m2+m4,       m1-m2+m3+m6]]
    
    So the factors are:
    U_2[i,j,r]: coefficient of A[i,j] in term r
    V_2[j,k,r]: coefficient of B[j,k] in term r
    W_2[k,i,r]: coefficient added to C[i,k] from term r
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # 2x2 Strassen factors
    # U_2[i*2+j, r]: coeff of A[i,j] in m_r
    U2 = np.zeros((4, 7))
    # r=0: m1 = (a11+a22)*(b11+b22)
    U2[0, 0] = 1  # a11
    U2[3, 0] = 1  # a22
    # r=1: m2 = (a21+a22)*b11
    U2[2, 1] = 1  # a21
    U2[3, 1] = 1  # a22
    # r=2: m3 = a11*(b12-b22)
    U2[0, 2] = 1  # a11
    # r=3: m4 = a22*(b21-b11)
    U2[3, 3] = 1  # a22
    # r=4: m5 = (a11+a12)*b22
    U2[0, 4] = 1  # a11
    U2[1, 4] = 1  # a12
    # r=5: m6 = (a21-a11)*(b11+b12)
    U2[2, 5] = 1   # a21
    U2[0, 5] = -1  # -a11
    # r=6: m7 = (a12-a22)*(b21+b22)
    U2[1, 6] = 1   # a12
    U2[3, 6] = -1  # -a22
    
    # V_2[j*2+k, r]: coeff of B[j,k] in m_r
    V2 = np.zeros((4, 7))
    # r=0: (b11+b22)
    V2[0, 0] = 1  # b11
    V2[3, 0] = 1  # b22
    # r=1: b11
    V2[0, 1] = 1
    # r=2: (b12-b22)
    V2[1, 2] = 1   # b12
    V2[3, 2] = -1  # -b22
    # r=3: (b21-b11)
    V2[2, 3] = 1   # b21
    V2[0, 3] = -1  # -b11
    # r=4: b22
    V2[3, 4] = 1
    # r=5: (b11+b12)
    V2[0, 5] = 1  # b11
    V2[1, 5] = 1  # b12
    # r=6: (b21+b22)
    V2[2, 6] = 1  # b21
    V2[3, 6] = 1  # b22
    
    # W_2[k*2+i, r]: coeff added to C[i,k] from m_r
    # C = [[m1+m4-m5+m7, m3+m5],
    #      [m2+m4,       m1-m2+m3+m6]]
    W2 = np.zeros((4, 7))
    # C[0,0] = m1+m4-m5+m7 -> W2[0,0]=1 (k=0,i=0)
    W2[0, 0] = 1   # m1
    W2[0, 3] = 1   # m4
    W2[0, 4] = -1  # -m5
    W2[0, 6] = 1   # m7
    # C[0,1] = m3+m5 -> W2[1,0]=1 (k=0,i=1)
    W2[1, 2] = 1   # m3
    W2[1, 4] = 1   # m5
    # C[1,0] = m2+m4 -> W2[2,0]=1 (k=1,i=0)
    W2[2, 1] = 1   # m2
    W2[2, 3] = 1   # m4
    # C[1,1] = m1-m2+m3+m6 -> W2[3,0]=1 (k=1,i=1)
    W2[3, 0] = 1   # m1
    W2[3, 1] = -1  # -m2
    W2[3, 2] = 1   # m3
    W2[3, 5] = 1   # m6
    
    # Verify 2x2 case
    U2_4 = U2.reshape(2, 2, 7)
    V2_4 = V2.reshape(2, 2, 7)
    W2_4 = W2.reshape(2, 2, 7)
    recon2 = np.einsum('ijr,jkr,krir->ijk', U2_4, V2_4, W2_4)
    assert np.max(np.abs(recon2 - 1)) < 1e-10, "2x2 Strassen verification failed"
    
    # Now construct 4x4 from 2x2 using Kronecker product structure.
    # The 4x4 tensor T4[(i,j),(j,k),(k,i)] = 1 for all i,j,k in {0,1,2,3}
    # This equals the Kronecker product of the 2x2 tensor with itself.
    # 
    # Partition 4x4 into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Axy is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B
    #
    # C[bl_i, bl_k, i2, k2] = sum over bl_j, j2 of A[bl_i, bl_j, i2, j2] * B[bl_j, bl_k, j2, k2]
    #
    # Using the rank-7 decomposition for each 2x2 block:
    # A[bl_i, i2] * B[bl_j, j2] is decomposed via 7 terms
    # But we need the full 4x4 product, which is a Kronecker structure.
    #
    # The rank-49 decomposition: for each pair (r1, r2) of Strassen terms:
    # The term corresponds to the Kronecker product of the 2x2 term r1 and r2.
    #
    # U4[i,j,r1*7+r2]: U2[i//2, j//2, r1] * U2[i%2, j%2, r2]  -- wait, this isn't quite right.
    #
    # Actually, the 4x4 multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # Then C[2*i1+i0, 2*k1+k0] = sum_{j1,j0} A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    #
    # = sum_{j1,j0} A_{i1,j1,i0,j0} * B_{j1,k1,j0,k0}
    #
    # This is the Kronecker product structure. The 4x4 tensor is T4 = T2 ⊗ T2.
    # The rank of T2 ⊗ T2 is at most rank(T2)^2 = 49.
    #
    # The factors for T2 ⊗ T2:
    # U4[(i1*2+i0), (j1*2+j0), r1*7+r2] = U2[i1, j1, r1] * U2[i0, j0, r2]
    # V4[(j1*2+j0), (k1*2+k0), r1*7+r2] = V2[j1, k1, r1] * V2[j0, k0, r2]
    # W4[(k1*2+k0), (i1*2+i0), r1*7+r2] = W2[k1, i1, r1] * W2[k0, i0, r2]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i0 in range(2):
                            for j0 in range(2):
                                for k0 in range(2):
                                    i = i1 * 2 + i0
                                    j = j1 * 2 + j0
                                    k = k1 * 2 + k0
                                    U[i * 4 + j, r] = U2[i1 * 2 + j1, r1] * U2[i0 * 2 + j0, r2]
                                    V[j * 4 + k, r] = V2[j1 * 2 + k1, r1] * V2[j0 * 2 + k0, r2]
                                    W[k * 4 + i, r] = W2[k1 * 2 + i1, r1] * W2[k0 * 2 + i0, r2]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try rank-49 Strassen first
    try:
        sol49 = _construct_rank49_strassen()
        U49 = np.array(sol49['U'])
        V49 = np.array(sol49['V'])
        W49 = np.array(sol49['W'])
        
        if _verify(U49, V49, W49):
            return sol49
    except Exception as e:
        print(f"Rank-49 failed: {e}")
    
    # Fallback: schoolbook rank 64
    if time.time() < deadline:
        return _schoolbook_64()
    
    return _schoolbook_64()

# EVOLVE-BLOCK-END
