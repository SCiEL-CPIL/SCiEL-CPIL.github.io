"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-10)

def _strassen_2x2x2():
    """Return Strassen decomposition of M_{2,2,2} as (U, V, W) with rank 7."""
    # n=2, m=2, p=2
    # U: shape (4, 7), indexed by i*2+j
    # V: shape (4, 7), indexed by j*2+k
    # W: shape (4, 7), indexed by k*2+i
    
    # Strassen's 7 rank-1 terms:
    # Each term is (u_vec, v_vec, w_vec) where each vec is in R^4
    
    # Term 1: (A00+A11)(B00+B11)
    # A00 is at U index 0*2+0=0, A11 is at U index 1*2+1=3
    # B00 is at V index 0*2+0=0, B11 is at V index 1*2+1=3
    # Contributes to C00 (W index 0*2+0=0) and C11 (W index 1*2+1=3)
    # But in tensor form, each rank-1 term has a single W vector.
    # The contribution to C00 and C11 means W has non-zero entries at 0 and 3.
    
    # Let me define the 7 terms directly:
    terms = []
    
    # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # Term 2: u=[0,1,0,1], v=[1,0,0,0], w=[0,1,0,0]
    terms.append(([0,1,0,1], [1,0,0,0], [0,1,0,0]))
    # Term 3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,0,1,0]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,0]))
    # Term 4: u=[0,0,0,1], v=[0,0,1,-1], w=[0,1,0,0])
    terms.append(([0,0,0,1], [0,0,1,-1], [0,1,0,0]))
    # Term 5: u=[1,1,0,0], v=[0,0,0,1], w=[0,0,1,0]
    terms.append(([1,1,0,0], [0,0,0,1], [0,0,1,0]))
    # Term 6: u=[1,0,-1,0], v=[1,0,1,0], w=[0,1,0,0]
    terms.append(([1,0,-1,0], [1,0,1,0], [0,1,0,0]))
    # Term 7: u=[0,1,0,-1], v=[0,1,0,1], w=[0,0,0,1]
    terms.append(([0,1,0,-1], [0,1,0,1], [0,0,0,1]))
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for c, (u, v, w) in enumerate(terms):
        U[:, c] = u
        V[:, c] = v
        W[:, c] = w
    
    return U, V, W

def _schoolbook_1x2x3():
    """Return schoolbook decomposition of M_{1,2,3} as (U, V, W) with rank 6."""
    # n=1, m=2, p=3
    # U: shape (2, 6), indexed by i*2+j = j (since i=0)
    # V: shape (6, 6), indexed by j*3+k
    # W: shape (3, 6), indexed by k*1+i = k (since i=0)
    
    U = np.zeros((2, 6))
    V = np.zeros((6, 6))
    W = np.zeros((3, 6))
    
    # 6 terms: for j in {0,1}, k in {0,1,2}
    col = 0
    for j in range(2):
        for k in range(3):
            U[j, col] = 1  # i=0, so U index = 0*2+j = j
            V[j*3+k, col] = 1
            W[k, col] = 1  # i=0, so W index = k*1+0 = k
            col += 1
    
    return U, V, W

def _kronecker_decomp(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Compute Kronecker product decomposition."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            # Kronecker product of U1[:, r1_idx] and U2[:, r2_idx]
            # U1[:, r1_idx] is in R^{n1*m1}, U2[:, r2_idx] is in R^{n2*m2}
            # The Kronecker product is in R^{n1*n2 * m1*m2}
            # Index mapping: (i1*m1+j1, i2*m2+j2) -> (i1*n2+i2)*(m1*m2) + j1*m2+j2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            idx1 = i1*m1 + j1
                            idx2 = i2*m2 + j2
                            idx = (i1*n2 + i2)*(m1*m2) + j1*m2 + j2
                            U[idx, r_idx] = U1[idx1, r1_idx] * U2[idx2, r2_idx]
            
            # Similarly for V
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            idx1 = j1*p1 + k1
                            idx2 = j2*p2 + k2
                            idx = (j1*m2 + j2)*(p1*p2) + k1*p2 + k2
                            V[idx, r_idx] = V1[idx1, r1_idx] * V2[idx2, r2_idx]
            
            # Similarly for W
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            idx1 = k1*n1 + i1
                            idx2 = k2*n2 + i2
                            idx = (k1*p2 + k2)*(n1*n2) + i1*n2 + i2
                            W[idx, r_idx] = W1[idx1, r1_idx] * W2[idx2, r2_idx]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # Try Kronecker product approach: M_{2,4,6} = M_{2,2,2} ⊗ M_{1,2,3}
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _schoolbook_1x2x3()
    
    U, V, W = _kronecker_decomp(U1, V1, W1, 2, 2, 2, U2, V2, W2, 1, 2, 3)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
