"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strassen 2x2x2 factor matrices
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # M1 = (A11+A22)*(B11+B22) -> C11 + C22
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1
    
    # M2 = (A21+A22)*B11 -> C21
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[2, 1] = 1
    
    # M3 = A11*(B11-B21) -> C12 - C22
    U22[0, 2] = 1
    V22[0, 2] = 1; V22[2, 2] = -1
    W22[1, 2] = 1; W22[3, 2] = -1
    
    # M4 = A22*(B22-B12) -> -C11 + C12
    U22[3, 3] = 1
    V22[3, 3] = 1; V22[1, 3] = -1
    W22[0, 3] = -1; W22[1, 3] = 1
    
    # M5 = (A11+A12)*B22 -> C11 + C21
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = 1; W22[2, 4] = 1
    
    # M6 = (A21-A11)*(B11+B12) -> C22
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[3, 5] = 1
    
    # M7 = (A12-A22)*(B21+B22) -> C11 - C22
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = 1; W22[3, 6] = -1
    
    # Place 6 non-overlapping 2x2x2 Strassen blocks
    # i0 in {0, 2}, j0 = 0, k0 in {0, 2, 4}
    blocks = []
    for i0 in [0, 2]:
        for k0 in [0, 2, 4]:
            blocks.append((i0, 0, k0))
    
    total_rank = len(blocks) * 7
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for bi, (i0, j0, k0) in enumerate(blocks):
        base_col = bi * 7
        for r in range(7):
            for ui in range(2):
                for uj in range(2):
                    u22_idx = ui * 2 + uj
                    if U22[u22_idx, r] != 0:
                        U[(i0 + ui) * m + (j0 + uj), base_col + r] = U22[u22_idx, r]
            
            for vj in range(2):
                for vk in range(2):
                    v22_idx = vj * 2 + vk
                    if V22[v22_idx, r] != 0:
                        V[(j0 + vj) * p + (k0 + vk), base_col + r] = V22[v22_idx, r]
            
            for wk in range(2):
                for wi in range(2):
                    w22_idx = wk * 2 + wi
                    if W22[w22_idx, r] != 0:
                        W[(k0 + wk) * n + (i0 + wi), base_col + r] = W22[w22_idx, r]
    
    # Verify the full decomposition
    def verify_decomp(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U, V, W, n, m, p):
        # Fall back to pure schoolbook
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
