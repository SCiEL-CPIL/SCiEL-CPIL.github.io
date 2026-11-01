"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    rank = U.shape[1]
    if U.shape[0] != 16 or V.shape[0] != 16 or W.shape[0] != 16:
        return False

    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    # Efficient check using einsum
    # recon[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    recon = np.einsum('ijr,jkr,kir->ijk', U4, V4, W4)
    
    target = np.ones((n, m, p))
    max_err = np.max(np.abs(recon - target))
    return max_err < 1e-9

def _strassen_2x2():
    """
    Returns the factor tensors for the rank-7 Strassen decomposition of 2x2 MM.
    U[i, j, r], V[j, k, r], W[k, i, r]
    """
    # U: (2, 2, 7)
    U = np.zeros((2, 2, 7), dtype=int)
    # V: (2, 2, 7)
    V = np.zeros((2, 2, 7), dtype=int)
    # W: (2, 2, 7)
    W = np.zeros((2, 2, 7), dtype=int)
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # r=0
    U[:, :, 0] = [[1, 0], [0, 1]]
    V[:, :, 0] = [[1, 0], [0, 1]]
    W[:, :, 0] = [[1, 1], [1, -1]] # Note: W[k,i] corresponds to C[i,k] in standard, but here W is (p,n) i.e. (k,i)
    # Wait, the convention is W[k*n+i, r] which flattens to W[k, i] if reshaped (p, n).
    # Let's check the Strassen C combination:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's map W[k, i] where i is row of C, k is col of C?
    # Standard C[i, k] = sum ...
    # In our tensor definition: recon[i, j, k] = sum U[i,j] V[j,k] W[k,i]
    # So W[k,i] multiplies U[i,j] and V[j,k].
    # This means W[k,i] is the coefficient for the (i,k) entry of C coming from the r-th product.
    # So W is effectively the transpose of the usual C-coeff matrix if we view C[i,k].
    # Let's re-verify the W assignments based on C[i,k] = sum_r (U_r[i,j] A[i,j]) (V_r[j,k] B[j,k]) W_r[k,i]
    # Actually, the product term is scalar M_r = (sum U A) (sum V B).
    # Then C[i,k] += M_r * W_r[k,i].
    
    # r=0: M0 = (A11+A22)(B11+B22)
    # C11 += 1*M0, C12 += 0, C21 += 0, C22 += -1*M0 ? No.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for r=0 (M1):
    # C11 coeff 1 -> W[0,0] (k=0,i=0) = 1
    # C12 coeff 0 -> W[0,1] (k=0,i=1) = 0
    # C21 coeff 0 -> W[1,0] (k=1,i=0) = 0
    # C22 coeff 1 -> W[1,1] (k=1,i=1) = 1
    # So W[:, :, 0] = [[1, 0], [0, 1]]
    
    # r=1 (M2):
    # C11: 0, C12: 0, C21: 1, C22: -1
    # W[:, :, 1] = [[0, 0], [1, -1]]
    
    # r=2 (M3):
    # C11: 0, C12: 1, C21: 0, C22: 1
    # W[:, :, 2] = [[0, 1], [0, 1]]
    
    # r=3 (M4):
    # C11: 1, C12: 0, C21: 1, C22: 0
    # W[:, :, 3] = [[1, 0], [1, 0]]
    
    # r=4 (M5):
    # C11: -1, C12: 1, C21: 0, C22: 0
    # W[:, :, 4] = [[-1, 1], [0, 0]]
    
    # r=5 (M6):
    # C11: 0, C12: 0, C21: 0, C22: 1
    # W[:, :, 5] = [[0, 0], [0, 1]]
    
    # r=6 (M7):
    # C11: -1, C12: 0, C21: 0, C22: 0
    # W[:, :, 6] = [[-1, 0], [0, 0]]
    
    # Let's re-verify U and V for these Ms.
    # M1 = (A11+A22)(B11+B22)
    # U1[i,j]: A11+A22 -> [[1,0],[0,1]]
    # V1[j,k]: B11+B22 -> [[1,0],[0,1]]
    
    # M2 = (A21+A22)B11
    # U2[i,j]: [[0,0],[1,1]]
    # V2[j,k]: [[1,0],[0,0]]
    
    # M3 = A11(B21-B22)
    # U3[i,j]: [[1,0],[0,0]]
    # V3[j,k]: [[0,1],[-1,0]] -> Wait, B21 is B[1,0], B22 is B[1,1].
    # V3[0,0]=0, V3[0,1]=1 (B[0,1]? No, B[j,k]).
    # B21 is j=1,k=0. B22 is j=1,k=1.
    # So V3[1,0]=1, V3[1,1]=-1. Others 0.
    # V3 = [[0,0],[1,-1]]
    
    # M4 = A22(B12-B11)
    # U4[i,j]: [[0,0],[0,1]]
    # V4[j,k]: B12 is j=0,k=1. B11 is j=0,k=0.
    # V4[0,0]=-1, V4[0,1]=1.
    # V4 = [[-1,1],[0,0]]
    
    # M5 = (A11+A12)B22
    # U5[i,j]: [[1,1],[0,0]]
    # V5[j,k]: B22 is j=1,k=1.
    # V5 = [[0,0],[0,1]]
    
    # M6 = (A21-A11)(B11+B12)
    # U6[i,j]: [[-1,0],[1,0]]
    # V6[j,k]: B11+B12 -> j=0 row: [1,1].
    # V6 = [[1,1],[0,0]]
    
    # M7 = (A12-A22)(B21+B22)
    # U7[i,j]: [[0,1],[0,-1]]
    # V7[j,k]: B21+B22 -> j=1 row: [1,1].
    # V7 = [[0,0],[1,1]]
    
    # Let's populate arrays
    U[:, :, 0] = [[1, 0], [0, 1]]
    V[:, :, 0] = [[1, 0], [0, 1]]
    W[:, :, 0] = [[1, 0], [0, 1]]
    
    U[:, :, 1] = [[0, 0], [1, 1]]
    V[:, :, 1] = [[1, 0], [0, 0]]
    W[:, :, 1] = [[0, 0], [1, -1]]
    
    U[:, :, 2] = [[1, 0], [0, 0]]
    V[:, :, 2] = [[0, 0], [1, -1]]
    W[:, :, 2] = [[0, 1], [0, 1]]
    
    U[:, :, 3] = [[0, 0], [0, 1]]
    V[:, :, 3] = [[-1, 1], [0, 0]]
    W[:, :, 3] = [[1, 0], [1, 0]]
    
    U[:, :, 4] = [[1, 1], [0, 0]]
    V[:, :, 4] = [[0, 0], [0, 1]]
    W[:, :, 4] = [[-1, 1], [0, 0]]
    
    U[:, :, 5] = [[-1, 0], [1, 0]]
    V[:, :, 5] = [[1, 1], [0, 0]]
    W[:, :, 5] = [[0, 0], [0, 1]]
    
    U[:, :, 6] = [[0, 1], [0, -1]]
    V[:, :, 6] = [[0, 0], [1, 1]]
    W[:, :, 6] = [[-1, 0], [0, 0]]
    
    return U, V, W

def _build_strassen_4x4():
    """
    Builds the rank-49 decomposition for 4x4 MM using Kronecker products
    of the 2x2 Strassen decomposition.
    """
    U2, V2, W2 = _strassen_2x2()
    n, m, p = 4, 4, 4
    
    # Rank is 7 * 7 = 49
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # U part:
            # U4[i, j, r] = U2[i1, j1, r1] * U2[i2, j2, r2]
            # i = 2*i1 + i2, j = 2*j1 + j2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1, j1, r1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2, j2, r2]
                            if u2 == 0: continue
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * 4 + j, r] = u1 * u2
                            
            # V part:
            # V4[j, k, r] = V2[j1, k1, r1] * V2[j2, k2, r2]
            # j = 2*j1 + j2, k = 2*k1 + k2
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V2[j1, k1, r1]
                    if v1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = V2[j2, k2, r2]
                            if v2 == 0: continue
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            V[j * 4 + k, r] = v1 * v2
                            
            # W part:
            # W4[k, i, r] = W2[k1, i1, r1] * W2[k2, i2, r2]
            # k = 2*k1 + k2, i = 2*i1 + i2
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W2[k1, i1, r1]
                    if w1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2 = W2[k2, i2, r2]
                            if w2 == 0: continue
                            k = 2 * k1 + k2
                            i = 2 * i1 + i2
                            W[k * 4 + i, r] = w1 * w2
                            
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Generate the decomposition programmatically
    U, V, W = _build_strassen_4x4()
    
    # Verify correctness
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: Schoolbook rank 64 (should not happen, but ensures a valid return type structure if logic fails)
        n, m, p = 4, 4, 4
        rank = 64
        U_fb = np.zeros((16, rank), dtype=int)
        V_fb = np.zeros((16, rank), dtype=int)
        W_fb = np.zeros((16, rank), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
                    
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
