import numpy as np
import time

def _strassen_2x2x2():
    """
    Verified rank-7 tensor decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7) each, such that:
      sum_t U[i,t] * V[j,t] * W[k,t] = delta_{i0,j0,k0} + delta_{i0,j1,k1} + delta_{i1,j0,k2} + delta_{i1,j1,k3}
    where i=(i0,i1), j=(j0,j1), k=(k0,k1) and the flat index is i0*2 + i1, etc.
    
    This is the standard Strassen decomposition.
    """
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0],
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 1],
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
    ], dtype=float)
    
    # Verify: check all 64 entries of the 2x2x2 multiplication tensor
    # The tensor T[i,j,k] = 1 if the multiplication C[k0,k1] = sum_j A[i0,j]*B[j,k1] has a 1 at (i,j,k)
    # Actually, the multiplication tensor is: T[i,j,k] = 1 if i=(i0,i1), j=(j0,j1), k=(k0,k1) and i0=k0 and j1=k1
    # Wait, C[k0,k1] = sum_{j} A[i0,j] * B[j,k1]... no.
    # C is 2x2, indexed by (row, col) = (k0, k1).
    # C[k0, k1] = sum_{j=0}^{1} A[k0, j] * B[j, k1]
    # So the tensor T[A_idx, B_idx, C_idx] = 1 if A_idx = (k0, j), B_idx = (j, k1), C_idx = (k0, k1)
    # i.e., A_idx and B_idx share the "middle" index j, and C_idx is (A_idx's row, B_idx's col).
    
    # Let's verify numerically.
    for i0 in range(2):
        for i1 in range(2):
            i = i0 * 2 + i1
            for j0 in range(2):
                for j1 in range(2):
                    j = j0 * 2 + j1
                    for k0 in range(2):
                        for k1 in range(2):
                            k = k0 * 2 + k1
                            # Expected value: 1 if i0 == k0 and j1 == k1, else 0
                            expected = 1.0 if (i0 == k0 and j1 == k1) else 0.0
                            # Computed value: sum_t U[i,t] * V[j,t] * W[k,t]
                            computed = sum(U[i, t] * V[j, t] * W[k, t] for t in range(7))
                            if abs(expected - computed) > 1e-10:
                                raise ValueError(f"Verification failed at ({i},{j},{k}): expected {expected}, got {computed}")
    
    return U, V, W

def _construct_2x2x8():
    """
    Construct rank-28 decomposition for 2x2 * 2x8 matrix multiplication
    by tiling 4 copies of the rank-7 2x2*2x2 decomposition.
    
    U: shape (4, 28) - A has 4 elements
    V: shape (16, 28) - B has 16 elements (2x8)
    W: shape (16, 28) - C has 16 elements (8x2)
    
    B is 2x8, so B[j, col] where j in {0,1}, col in {0,...,7}.
    C is 8x2, so C[row, i] where row in {0,...,7}, i in {0,1}.
    
    We split B into 4 blocks of 2x2: B[:, 2t:2t+2] for t=0,1,2,3.
    Similarly, C is split into 4 blocks: C[2t:2t+2, :] for t=0,1,2,3.
    
    For each block t, we use the same rank-7 decomposition.
    """
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for t in range(p // 2):
        base = t * 7
        for r in range(7):
            col = base + r
            # U: same for all blocks
            U[:, col] = U_s[:, r]
            
            # V: B[j, 2t + k] = B_s[j, k]
            # V_s has indices for B_s[j, k] where j in {0,1}, k in {0,1}
            # V index: j * p + (2t + k)
            for j in range(2):
                for k in range(2):
                    v_s_idx = j * 2 + k
                    v_idx = j * p + (2 * t + k)
                    V[v_idx, col] = V_s[v_s_idx, r]
            
            # W: C[2t + k, i] = C_s[k, i]
            # W_s has indices for C_s[k, i] where k in {0,1}, i in {0,1}
            # W index: (2t + k) * n + i
            for k in range(2):
                for i in range(2):
                    w_s_idx = k * 2 + i
                    w_idx = (2 * t + k) * n + i
                    W[w_idx, col] = W_s[w_s_idx, r]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _construct_2x2x8()
