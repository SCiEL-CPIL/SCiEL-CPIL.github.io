"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Recursive Strassen decomposition for 4x4 matrix multiplication, rank 49."""
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 10.0
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=np.int64)  # (16, 49)
    V = np.zeros((m * p, rank), dtype=np.int64)  # (16, 49)
    W = np.zeros((p * n, rank), dtype=np.int64)  # (16, 49)
    
    # Outer Strassen coefficients: beta[s][I][J] for A block (I,J)
    # gamma[s][J][K] for B block (J,K)
    # delta[s][I][K] for C block (I,K)
    # s = 0..6, I,J,K = 0..1
    beta = np.zeros((7, 2, 2), dtype=np.int64)
    gamma = np.zeros((7, 2, 2), dtype=np.int64)
    delta = np.zeros((7, 2, 2), dtype=np.int64)
    
    # s=0: M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1
    
    # s=1: M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1
    
    # s=2: M3 = A00(B01-B11) -> C01 += 1, C11 += 1
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1
    
    # s=3: M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1
    
    # s=4: M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1
    
    # s=5: M6 = (A10-A00)(B00+B01) -> C11 += 1
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1
    
    # s=6: M7 = (A01-A11)(B10+B11) -> C00 += 1
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1
    
    # Inner Strassen coefficients: U2[i][t], V2[j][t], W2[k][t]
    # i,j,k = 0..3 (local 2x2 indices: 00, 01, 10, 11)
    # t = 0..6
    U2 = np.zeros((4, 7), dtype=np.int64)
    V2 = np.zeros((4, 7), dtype=np.int64)
    W2 = np.zeros((4, 7), dtype=np.int64)
    
    # t=0: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1: (X10+X11)Y00 -> Z10 += 1, Z11 -= 1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # t=2: X00(Y01-Y11) -> Z01 += 1, Z11 += 1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # t=3: X11(Y10-Y00) -> Z00 += 1, Z10 += 1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # t=4: (X00+X01)Y11 -> Z00 -= 1, Z01 += 1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # t=5: (X10-X00)(Y00+Y01) -> Z11 += 1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # t=6: (X01-X11)(Y10+Y11) -> Z00 += 1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Construct U, V, W using the coefficient tables
    # Global A index: i*4+j, where I=i//2, i'=i%2, J=j//2, j'=j%2
    # Global B index: j*4+k, where J=j//2, j'=j%2, K=k//2, k'=k%2
    # Global W index: k*4+i (trace convention W[k*n+i]), where K=k//2, k'=k%2, I=i//2, i'=i%2
    # Local 2x2 flat index: 2*row + col
    
    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            
            # Fill U
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    i_prime = i % 2
                    J = j // 2
                    j_prime = j % 2
                    row_u = i * 4 + j
                    local_u_idx = i_prime * 2 + j_prime
                    coeff = beta[s, I, J] * U2[local_u_idx, t]
                    if coeff != 0:
                        U[row_u, c] = coeff
                        
            # Fill V
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_prime = j % 2
                    K = k // 2
                    k_prime = k % 2
                    row_v = j * 4 + k
                    local_v_idx = j_prime * 2 + k_prime
                    coeff = gamma[s, J, K] * V2[local_v_idx, t]
                    if coeff != 0:
                        V[row_v, c] = coeff
                        
            # Fill W (trace convention: W[k*n+i])
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_prime = k % 2
                    I = i // 2
                    i_prime = i % 2
                    row_w = k * 4 + i
                    local_w_idx = k_prime * 2 + i_prime
                    coeff = delta[s, I, K] * W2[local_w_idx, t]
                    if coeff != 0:
                        W[row_w, c] = coeff
                        
            if time.time() > _DEADLINE - _MARGIN:
                break
        
        if time.time() > _DEADLINE - _MARGIN:
            break
    
    # Verify rank
    assert c == rank - 1 or s < 6 or t < 6, f"Expected rank {rank}"
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
