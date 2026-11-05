import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    # Start with the known rank-55 decomposition
    # Strategy: 5 blocks of size 2 in p-dimension, each using Strassen-like 7-term decomposition for 2x2x2
    # Plus 20 schoolbook terms for the remaining 2x1x10 part
    # Total: 5*7 + 20 = 55

    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Strassen coefficients for 2x2x2 matrix multiplication
    # U coefficients: U[i*m+j, r] where i in [0,1], j in [0,1]
    # V coefficients: V[j*p+k, r] where j in [0,1], k in [0,1]
    # W coefficients: W[k*n+i, r] where i in [0,1], k in [0,1]
    
    # Standard Strassen coefficients (7 terms)
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B00)
    # M3 = A11(B01-B11)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M5 + M6
    
    L = [  # U coefficients (A matrix part)
        {0: 1, 3: 1},   # i=0,j=0 and i=1,j=1
        {2: 1, 3: 1},   # i=1,j=0 and i=1,j=1
        {0: 1},          # i=0,j=0
        {3: 1},          # i=1,j=1
        {0: 1, 1: 1},   # i=0,j=0 and i=0,j=1
        {2: 1, 0: -1},  # i=1,j=0 and i=0,j=0
        {1: 1, 3: -1}   # i=0,j=1 and i=1,j=1
    ]
    R = [  # V coefficients (B matrix part)
        {0: 1, 3: 1},   # j=0,k=0 and j=1,k=1
        {0: 1},          # j=0,k=0
        {1: 1, 3: -1},  # j=1,k=0 and j=1,k=1
        {2: 1, 0: -1},  # j=1,k=1 and j=0,k=0
        {3: 1},          # j=1,k=1
        {0: 1, 1: 1},   # j=0,k=0 and j=0,k=1
        {2: 1, 3: 1}    # j=1,k=1 and j=1,k=0
    ]
    Wc = [  # W coefficients (C matrix part)
        {0: 1, 3: 1},   # k=0,i=0 and k=1,i=1
        {2: 1, 3: -1},  # k=1,i=0 and k=1,i=1
        {1: 1, 3: 1},   # k=0,i=0 and k=1,i=1
        {0: 1, 2: 1},   # k=0,i=0 and k=1,i=0
        {1: 1, 0: -1},  # k=0,i=0 and k=0,i=1
        {3: 1},          # k=1,i=1
        {0: 1}           # k=0,i=0
    ]
    
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1

    # Add 20 schoolbook terms for the 2x1x10 part (j=2, i in [0,1], k in [0,10])
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Verify the decomposition is correct using vectorized numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def check_valid(U_mat, V_mat, W_mat):
        """Check if decomposition is valid: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        # Reshape for broadcasting: U_mat[i*m+j, r], V_mat[j*p+k, r], W_mat[k*n+i, r]
        # Create tensor of shape (n, m, p, rank)
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not check_valid(U_np, V_np, W_np):
        # Fallback: return a simpler valid decomposition
        # Use all schoolbook terms: rank = 60
        rank = 60
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

    # Try to reduce rank by attempting to eliminate redundant terms (vectorized)
    def try_remove_term(U_mat, V_mat, W_mat, term_idx):
        """Check if removing term_idx still gives valid decomposition"""
        U_r = np.delete(U_mat, term_idx, axis=1)
        V_r = np.delete(V_mat, term_idx, axis=1)
        W_r = np.delete(W_mat, term_idx, axis=1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return None
        return (U_r, V_r, W_r)

    best_rank = rank
    # Try removing each term, keep trying until no more can be removed
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        current_rank = U_np.shape[1]
        for r in range(current_rank):
            if time.time() > deadline:
                break
            result = try_remove_term(U_np, V_np, W_np, r)
            if result is not None:
                U_np, V_np, W_np = result
                rank = U_np.shape[1]
                best_rank = rank
                removed_any = True
                break

    # Convert back to lists
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()

    # Ensure entries are integers or half-integers
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2

    return {"U": U_out, "V": V_out, "W": W_out}
