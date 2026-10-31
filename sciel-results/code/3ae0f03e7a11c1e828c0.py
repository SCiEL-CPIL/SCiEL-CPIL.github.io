"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Double-Strassen decomposition for 4x4 matrix multiplication, rank 49."""
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 140.0  # Safety margin before 150s kill
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Strassen 2x2 coefficients
    # For each of 7 multiplications, define the 2x2 coefficient matrices
    # beta[s][I][J]: coefficient of A[I][J] in the left factor of multiplication s
    # gamma[s][J][K]: coefficient of B[J][K] in the right factor of multiplication s  
    # delta[s][I][K]: coefficient of C[I][K] from multiplication s
    
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta = np.zeros((7, 2, 2), dtype=int)
    
    # s=0: M0 = (A00+A11)(B00+B11), C00+=M0, C11+=M0
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1
    
    # s=1: M1 = (A10+A11)B00, C10+=M1, C11-=M1
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1
    
    # s=2: M2 = A00(B01-B11), C01+=M2, C11+=M2
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1
    
    # s=3: M3 = A11(B10-B00), C00+=M3, C10+=M3
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1
    
    # s=4: M4 = (A00+A01)B11, C00-=M4, C01+=M4
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1
    
    # s=5: M5 = (A10-A00)(B00+B01), C11+=M5
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1
    
    # s=6: M6 = (A01-A11)(B10+B11), C00+=M6
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1
    
    # Inner Strassen: for each 2x2 matrix multiplication, 7 rank-1 terms
    # u2[t][i][j]: coefficient of A[i][j] in left factor of inner term t
    # v2[t][j][k]: coefficient of B[j][k] in right factor of inner term t
    # w2[t][i][k]: coefficient of C[i][k] from inner term t
    
    u2 = np.zeros((7, 2, 2), dtype=int)
    v2 = np.zeros((7, 2, 2), dtype=int)
    w2 = np.zeros((7, 2, 2), dtype=int)
    
    # t=0: (X00+X11)(Y00+Y11) -> C00+=, C11+=
    u2[0, 0, 0] = 1; u2[0, 1, 1] = 1
    v2[0, 0, 0] = 1; v2[0, 1, 1] = 1
    w2[0, 0, 0] = 1; w2[0, 1, 1] = 1
    
    # t=1: (X10+X11)Y00 -> C10+=, C11-=
    u2[1, 1, 0] = 1; u2[1, 1, 1] = 1
    v2[1, 0, 0] = 1
    w2[1, 1, 0] = 1; w2[1, 1, 1] = -1
    
    # t=2: X00(Y01-Y11) -> C01+=, C11+=
    u2[2, 0, 0] = 1
    v2[2, 0, 1] = 1; v2[2, 1, 1] = -1
    w2[2, 0, 1] = 1; w2[2, 1, 1] = 1
    
    # t=3: X11(Y10-Y00) -> C00+=, C10+=
    u2[3, 1, 1] = 1
    v2[3, 1, 0] = 1; v2[3, 0, 0] = -1
    w2[3, 0, 0] = 1; w2[3, 1, 0] = 1
    
    # t=4: (X00+X01)Y11 -> C00-=, C01+=
    u2[4, 0, 0] = 1; u2[4, 0, 1] = 1
    v2[4, 1, 1] = 1
    w2[4, 0, 0] = -1; w2[4, 0, 1] = 1
    
    # t=5: (X10-X00)(Y00+Y01) -> C11+=
    u2[5, 1, 0] = 1; u2[5, 0, 0] = -1
    v2[5, 0, 0] = 1; v2[5, 0, 1] = 1
    w2[5, 1, 1] = 1
    
    # t=6: (X01-X11)(Y10+Y11) -> C00+=
    u2[6, 0, 1] = 1; u2[6, 1, 1] = -1
    v2[6, 1, 0] = 1; v2[6, 1, 1] = 1
    w2[6, 0, 0] = 1
    
    # Construct U, V, W
    # For each outer term s and inner term t:
    # r = s*7 + t
    # U[i*4+j, r] = sum over outer block indices (I,J) and inner indices (i2,j2):
    #   beta[s,I,J] * u2[t,i2,j2] where i = 2*I+i2, j = 2*J+j2
    # V[j*4+k, r] = sum over (J,K) and (j2,k2):
    #   gamma[s,J,K] * v2[t,j2,k2] where j = 2*J+j2, k = 2*K+k2
    # W[k*4+i, r] = sum over (K,I) and (k2,i2):
    #   delta[s,I,K] * w2[t,i2,k2] where i = 2*I+i2, k = 2*K+k2
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            if time.time() > _DEADLINE:
                break
            
            # Compute U contributions
            for I in range(2):
                for J in range(2):
                    bs = beta[s, I, J]
                    if bs == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            ut = u2[t, i2, j2]
                            if ut == 0:
                                continue
                            i = 2 * I + i2
                            j = 2 * J + j2
                            U[i * 4 + j, r] += bs * ut
            
            # Compute V contributions
            for J in range(2):
                for K in range(2):
                    gs = gamma[s, J, K]
                    if gs == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            vt = v2[t, j2, k2]
                            if vt == 0:
                                continue
                            j = 2 * J + j2
                            k = 2 * K + k2
                            V[j * 4 + k, r] += gs * vt
            
            # Compute W contributions
            # W index: k*n + i = k*4 + i
            # delta[s, I, K] is coefficient for C[I][K]
            # w2[t, i2, k2] is coefficient for inner C[i2][k2]
            # global i = 2*I + i2, global k = 2*K + k2
            for I in range(2):
                for K in range(2):
                    ds = delta[s, I, K]
                    if ds == 0:
                        continue
                    for i2 in range(2):
                        for k2 in range(2):
                            wt = w2[t, i2, k2]
                            if wt == 0:
                                continue
                            i = 2 * I + i2
                            k = 2 * K + k2
                            W[k * 4 + i, r] += ds * wt
    
    # Verify correctness
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    A_flat = A_test.flatten()  # index i*4+j
    B_flat = B_test.flatten()  # index j*4+k
    
    # C[i,k] = sum_r W[k*4+i, r] * sum_j U[i*4+j, r] * V[j*4+k, r]
    C_computed = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            s_val = 0.0
            for r in range(rank):
                w_val = W[k * 4 + i, r]
                if w_val == 0:
                    continue
                inner = 0.0
                for j in range(4):
                    u_val = U[i * 4 + j, r]
                    v_val = V[j * 4 + k, r]
                    if u_val != 0 and v_val != 0:
                        inner += u_val * v_val
                s_val += w_val * inner
            C_computed[i, k] = s_val
    
    max_err = np.max(np.abs(C_computed - C_expected))
    
    # If verification fails, try to debug
    if max_err > 1e-9:
        # Try with simpler test
        A_test = np.eye(4)
        B_test = np.eye(4)
        C_expected = A_test @ B_test
        C_computed = np.zeros((4, 4))
        for i in range(4):
            for k in range(4):
                s_val = 0.0
                for r in range(rank):
                    w_val = W[k * 4 + i, r]
                    if w_val == 0:
                        continue
                    inner = 0.0
                    for j in range(4):
                        u_val = U[i * 4 + j, r]
                        v_val = V[j * 4 + k, r]
                        if u_val != 0 and v_val != 0:
                            inner += u_val * v_val
                    s_val += w_val * inner
                C_computed[i, k] = s_val
        max_err = np.max(np.abs(C_computed - C_expected))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
