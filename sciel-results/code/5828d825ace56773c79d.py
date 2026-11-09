import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 120

    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (12, rank)

    # Indices:
    # U[i*m + k] where i in 0..n-1, k in 0..m-1
    # V[k*p + j] where k in 0..m-1, j in 0..p-1
    # W[j*n + i] where j in 0..p-1, i in 0..n-1

    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_solution(U, V, W):
        R = U.shape[1]
        D = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(D, T, atol=1e-9):
            return False
        for mat in [U, V, W]:
            for val in mat.flatten():
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11) * (B00 + B11) -> C00 += M1, C11 -= M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1
        
        # M2 = (A10 + A11) * B00 -> C10 += M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00 * (B01 - B11) -> C01 += M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A11 * (B10 - B00) -> C11 += M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # M5 = (A00 + A01) * B11 -> C01 += M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01) -> C10 -= M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = -1
        
        # M7 = (A01 - A11) * (B10 + B11) -> C00 += M7, C11 -= M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1
        
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        U[1, 1] = 1
        V[2, 1] = 1
        W[0, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1
        W[1, 2] = 1
        
        U[1, 3] = 1
        V[3, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    def build_strassen_44():
        U_full = np.zeros((n * m, 44))
        V_full = np.zeros((m * p, 44))
        W_full = np.zeros((p * n, 44))
        
        col = 0
        
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        U_b, V_b, W_b = strassen_2x2x2()
        for r in range(7):
            for a_loc in range(4):
                i_loc = a_loc // 2
                k_loc = a_loc % 2
                a_glob = i_loc * m + k_loc
                U_full[a_glob, col + r] += U_b[a_loc, r]
            for b_loc in range(4):
                k_loc = b_loc // 2
                j_loc = b_loc % 2
                b_glob = k_loc * p + j_loc
                V_full[b_glob, col + r] += V_b[b_loc, r]
            for c_loc in range(4):
                i_loc = c_loc // 2
                j_loc = c_loc % 2
                c_glob = j_loc * n + i_loc
                W_full[c_glob, col + r] += W_b[c_loc, r]
        col += 7
        
        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        U_b, V_b, W_b = strassen_2x2x2()
        for r in range(7):
            for a_loc in range(4):
                i_loc = a_loc // 2
                k_loc = a_loc % 2
                a_glob = i_loc * m + (2 + k_loc)
                U_full[a_glob, col + r] += U_b[a_loc, r]
            for b_loc in range(4):
                k_loc = b_loc // 2
                j_loc = b_loc % 2
                b_glob = (2 + k_loc) * p + j_loc
                V_full[b_glob, col + r] += V_b[b_loc, r]
            for c_loc in range(4):
                i_loc = c_loc // 2
                j_loc = c_loc % 2
                c_glob = j_loc * n + i_loc
                W_full[c_glob, col + r] += W_b[c_loc, r]
        col += 7
        
        # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        U_b, V_b, W_b = strassen_2x2x2()
        for r in range(7):
            for a_loc in range(4):
                i_loc = a_loc // 2
                k_loc = a_loc % 2
                a_glob = i_loc * m + k_loc
                U_full[a_glob, col + r] += U_b[a_loc, r]
            for b_loc in range(4):
                k_loc = b_loc // 2
                j_loc = b_loc % 2
                b_glob = k_loc * p + (2 + j_loc)
                V_full[b_glob, col + r] += V_b[b_loc, r]
            for c_loc in range(4):
                i_loc = c_loc // 2
                j_loc = c_loc % 2
                c_glob = (2 + j_loc) * n + i_loc
                W_full[c_glob, col + r] += W_b[c_loc, r]
        col += 7
        
        # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        U_b, V_b, W_b = strassen_2x2x2()
        for r in range(7):
            for a_loc in range(4):
                i_loc = a_loc // 2
                k_loc = a_loc % 2
                a_glob = i_loc * m + (2 + k_loc)
                U_full[a_glob, col + r] += U_b[a_loc, r]
            for b_loc in range(4):
                k_loc = b_loc // 2
                j_loc = b_loc % 2
                b_glob = (2 + k_loc) * p + (2 + j_loc)
                V_full[b_glob, col + r] += V_b[b_loc, r]
            for c_loc in range(4):
                i_loc = c_loc // 2
                j_loc = c_loc % 2
                c_glob = (2 + j_loc) * n + i_loc
                W_full[c_glob, col + r] += W_b[c_loc, r]
        col += 7
        
        # Block 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
        U_b, V_b, W_b = schoolbook_1x2x2()
        for r in range(4):
            for a_loc in range(2):
                k_loc = a_loc
                a_glob = 2 * m + k_loc
                U_full[a_glob, col + r] += U_b[a_loc, r]
            for b_loc in range(4):
                k_loc = b_loc // 2
                j_loc = b_loc % 2
                b_glob = k_loc * p + j_loc
                V_full[b_glob, col + r] += V_b[b_loc, r]
            for c_loc in range(2):
                j_loc = c_loc
                c_glob = j_loc * n + 2
                W_full[c_glob, col + r] += W_b[c_loc, r]
        col += 4
        
        # Block 6: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
        U_b, V_b, W_b = schoolbook_1x2x2()
        for r in range(4):
            for a_loc in range(2):
                k_loc = a_loc
                a_glob = 2 * m + (2 + k_loc)
                U_full[a_glob, col + r] += U_b[a_loc, r]
            for b_loc in range(4):
                k_loc = b_loc // 2
                j_loc = b_loc % 2
                b_glob = (2 + k_loc) * p + j_loc
                V_full[b_glob, col + r] += V_b[b_loc, r]
            for c_loc in range(2):
                j_loc = c_loc
                c_glob = j_loc * n + 2
                W_full[c_glob, col + r] += W_b[c_loc, r]
        col += 4
        
        # Block 7: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
        U_b, V_b, W_b = schoolbook_1x2x2()
        for r in range(4):
            for a_loc in range(2):
                k_loc = a_loc
                a_glob = 2 * m + k_loc
                U_full[a_glob, col + r] += U_b[a_loc, r]
            for b_loc in range(4):
                k_loc = b_loc // 2
                j_loc = b_loc % 2
                b_glob = k_loc * p + (2 + j_loc)
                V_full[b_glob, col + r] += V_b[b_loc, r]
            for c_loc in range(2):
                j_loc = c_loc
                c_glob = (2 + j_loc) * n + 2
                W_full[c_glob, col + r] += W_b[c_loc, r]
        col += 4
        
        # Block 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
        U_b, V_b, W_b = schoolbook_1x2x2()
        for r in range(4):
            for a_loc in range(2):
                k_loc = a_loc
                a_glob = 2 * m + (2 + k_loc)
                U_full[a_glob, col + r] += U_b[a_loc, r]
            for b_loc in range(4):
                k_loc = b_loc // 2
                j_loc = b_loc % 2
                b_glob = (2 + k_loc) * p + (2 + j_loc)
                V_full[b_glob, col + r] += V_b[b_loc, r]
            for c_loc in range(2):
                j_loc = c_loc
                c_glob = (2 + j_loc) * n + 2
                W_full[c_glob, col + r] += W_b[c_loc, r]
        col += 4
        
        if verify_solution(U_full, V_full, W_full):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        return None

    # Try rank 44
    sol = build_strassen_44()
    if sol is not None:
        return sol

    # Fallback
    return get_schoolbook()
# EVOLVE-BLOCK-END
