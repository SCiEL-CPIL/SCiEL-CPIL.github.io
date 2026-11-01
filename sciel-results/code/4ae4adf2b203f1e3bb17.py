"""Find rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True

def construct():
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize matrices
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen factors for 2x2x2
    # U_strassen (4x7)
    U_s = np.zeros((4, 7))
    U_s[0, 0] = 1; U_s[3, 0] = 1  # m1: a+d
    U_s[2, 1] = 1; U_s[3, 1] = 1  # m2: c+d
    U_s[0, 2] = 1                 # m3: a
    U_s[3, 3] = 1                 # m4: d
    U_s[0, 4] = 1; U_s[1, 4] = 1  # m5: a+b
    U_s[2, 5] = 1; U_s[0, 5] = -1 # m6: c-a
    U_s[1, 6] = 1; U_s[3, 6] = -1 # m7: b-d
    
    # V_strassen (4x7) - Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V_s = np.zeros((4, 7))
    V_s[0, 0] = 1; V_s[3, 0] = 1  # m1: e+h
    V_s[0, 1] = 1                 # m2: e
    V_s[1, 2] = 1; V_s[3, 2] = -1 # m3: f-h
    V_s[2, 3] = 1; V_s[0, 3] = -1 # m4: g-e
    V_s[3, 4] = 1                 # m5: h
    V_s[0, 5] = 1; V_s[1, 5] = 1  # m6: e+f
    V_s[2, 6] = 1; V_s[3, 6] = 1  # m7: g+h
    
    # W_strassen (4x7) - Local C indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    # Note: W index is k*n + i. For 2x2, n=2.
    # Local C(i,k) -> W[k*2 + i]
    # C(0,0)->W[0], C(1,0)->W[1], C(0,1)->W[2], C(1,1)->W[3]
    W_s = np.zeros((4, 7))
    # p = C00 = m1 + m4 - m5 + m7 -> W[0]
    W_s[0, 0] = 1; W_s[0, 3] = 1; W_s[0, 4] = -1; W_s[0, 6] = 1
    # q = C01 = m3 + m5 -> W[2] (Wait, C(0,1) is i=0, k=1. Index 1*2+0=2)
    W_s[2, 2] = 1; W_s[2, 4] = 1
    # r = C10 = m2 + m4 -> W[1] (i=1, k=0. Index 0*2+1=1)
    W_s[1, 1] = 1; W_s[1, 3] = 1
    # s = C11 = m1 - m2 + m3 + m6 -> W[3] (i=1, k=1. Index 1*2+1=3)
    W_s[3, 0] = 1; W_s[3, 1] = -1; W_s[3, 2] = 1; W_s[3, 5] = 1
    
    # Block 1: Cols 0, 1 (Rank 7)
    U[:, 0:7] = U_s
    # V local rows 0,1,2,3 map to global rows:
    # Local 0 (B00) -> Global 0*7+0 = 0
    # Local 1 (B01) -> Global 0*7+1 = 1
    # Local 2 (B10) -> Global 1*7+0 = 7
    # Local 3 (B11) -> Global 1*7+1 = 8
    v_map_1 = [0, 1, 7, 8]
    for local_r, global_r in enumerate(v_map_1):
        V[global_r, 0:7] = V_s[local_r, :]
    # W local rows 0,1,2,3 map to global rows:
    # Local 0 (C00) -> Global 0*2+0 = 0
    # Local 1 (C10) -> Global 0*2+1 = 1
    # Local 2 (C01) -> Global 1*2+0 = 2
    # Local 3 (C11) -> Global 1*2+1 = 3
    w_map_1 = [0, 1, 2, 3]
    for local_r, global_r in enumerate(w_map_1):
        W[global_r, 0:7] = W_s[local_r, :]
        
    # Block 2: Cols 2, 3 (Rank 7)
    U[:, 7:14] = U_s
    # V local rows map to global:
    # Local 0 (B00) -> Global 0*7+2 = 2
    # Local 1 (B01) -> Global 0*7+3 = 3
    # Local 2 (B10) -> Global 1*7+2 = 9
    # Local 3 (B11) -> Global 1*7+3 = 10
    v_map_2 = [2, 3, 9, 10]
    for local_r, global_r in enumerate(v_map_2):
        V[global_r, 7:14] = V_s[local_r, :]
    # W local rows map to global:
    # Local 0 (C00) -> Global 2*2+0 = 4
    # Local 1 (C10) -> Global 2*2+1 = 5
    # Local 2 (C01) -> Global 3*2+0 = 6
    # Local 3 (C11) -> Global 3*2+1 = 7
    w_map_2 = [4, 5, 6, 7]
    for local_r, global_r in enumerate(w_map_2):
        W[global_r, 7:14] = W_s[local_r, :]
        
    # Block 3: Cols 4, 5 (Rank 7)
    U[:, 14:21] = U_s
    # V local rows map to global:
    # Local 0 (B00) -> Global 0*7+4 = 4
    # Local 1 (B01) -> Global 0*7+5 = 5
    # Local 2 (B10) -> Global 1*7+4 = 11
    # Local 3 (B11) -> Global 1*7+5 = 12
    v_map_3 = [4, 5, 11, 12]
    for local_r, global_r in enumerate(v_map_3):
        V[global_r, 14:21] = V_s[local_r, :]
    # W local rows map to global:
    # Local 0 (C00) -> Global 4*2+0 = 8
    # Local 1 (C10) -> Global 4*2+1 = 9
    # Local 2 (C01) -> Global 5*2+0 = 10
    # Local 3 (C11) -> Global 5*2+1 = 11
    w_map_3 = [8, 9, 10, 11]
    for local_r, global_r in enumerate(w_map_3):
        W[global_r, 14:21] = W_s[local_r, :]
        
    # Block 4: Col 6 (Rank 4, Schoolbook)
    # C_i6 = A_i0 B_06 + A_i1 B_16
    # Term 0: A_00 B_06 -> C_06
    # Term 1: A_01 B_16 -> C_06
    # Term 2: A_10 B_06 -> C_16
    # Term 3: A_11 B_16 -> C_16
    
    U[0, 21] = 1
    U[1, 22] = 1
    U[2, 23] = 1
    U[3, 24] = 1
    
    V[0*7+6, 21] = 1 # B_06
    V[1*7+6, 22] = 1 # B_16
    V[0*7+6, 23] = 1 # B_06
    V[1*7+6, 24] = 1 # B_16
    
    W[6*2+0, 21] = 1 # C_06
    W[6*2+0, 22] = 1 # C_06
    W[6*2+1, 23] = 1 # C_16
    W[6*2+1, 24] = 1 # C_16
    
    # Convert to list of lists of Fractions for exactness
    def to_fraction_matrix(mat):
        res = []
        for row in mat:
            new_row = []
            for val in row:
                if abs(val) < 1e-10:
                    new_row.append(0)
                else:
                    # Values are integers -1, 0, 1
                    new_row.append(int(round(val)))
            res.append(new_row)
        return res
        
    U_out = to_fraction_matrix(U)
    V_out = to_fraction_matrix(V)
    W_out = to_fraction_matrix(W)
    
    # Verify
    if not _verify_decomposition(U_out, V_out, W_out, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return {"U": U_out, "V": V_out, "W": W_out}
