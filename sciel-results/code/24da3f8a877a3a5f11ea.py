"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _tensor_from_decomp(U, V, W):
    """Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Check if decomposition exactly matches target tensor."""
    T_approx = _tensor_from_decomp(U, V, W)
    return np.max(np.abs(T_approx - T)) < tol

def _strassen_2x2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen (Rank 7).
    U: 4x7, V: 4x7, W: 4x7.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U corresponds to A, V to B, W to C.
    # We need to define the coefficients for each of the 7 terms.
    
    # Term 0: M1
    # A parts: A11(1), A22(1) -> U[0,0]=1, U[3,0]=1
    # B parts: B11(1), B22(1) -> V[0,0]=1, V[3,0]=1
    # C parts: C11(1), C22(1) -> W[0,0]=1, W[3,0]=1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # Term 1: M2
    # A parts: A21(1), A22(1) -> U[2,1]=1, U[3,1]=1
    # B parts: B11(1) -> V[0,1]=1
    # C parts: C21(1) -> W[2,1]=1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # Term 2: M3
    # A parts: A11(1) -> U[0,2]=1
    # B parts: B12(1), B22(-1) -> V[1,2]=1, V[3,2]=-1
    # C parts: C12(1) -> W[1,2]=1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1
    
    # Term 3: M4
    # A parts: A22(1) -> U[3,3]=1
    # B parts: B21(1), B11(-1) -> V[2,3]=1, V[0,3]=-1
    # C parts: C11(1), C21(1) -> W[0,3]=1, W[2,3]=1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # Term 4: M5
    # A parts: A11(1), A12(1) -> U[0,4]=1, U[1,4]=1
    # B parts: B22(1) -> V[3,4]=1
    # C parts: C11(-1), C12(1) -> W[0,4]=-1, W[1,4]=1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # Term 5: M6
    # A parts: A21(1), A11(-1) -> U[2,5]=1, U[0,5]=-1
    # B parts: B11(1), B12(1) -> V[0,5]=1, V[1,5]=1
    # C parts: C22(1) -> W[3,5]=1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # Term 6: M7
    # A parts: A12(1), A22(-1) -> U[1,6]=1, U[3,6]=-1
    # B parts: B21(1), B22(1) -> V[2,6]=1, V[3,6]=1
    # C parts: C11(1) -> W[0,6]=1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _m244_decomp():
    """
    Returns U, V, W for M_{2,4,4} (Rank 28).
    U: 8x28, V: 16x28, W: 8x28.
    """
    rank = 28
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))
    
    sU, sV, sW = _strassen_2x2x2()
    
    # 4 blocks
    # Block 0: A_L B_LL -> C_LL
    # A_L is rows 0,1 of A (2x4). B_LL is rows 0,1 of B (4x4) -> cols 0,1.
    # Wait, B is 4x4.
    # A is 2x4. A_L is A[:, 0:2], A_R is A[:, 2:4].
    # B is 4x4. B_LL is B[0:2, 0:2], B_LR is B[0:2, 2:4], B_RL is B[2:4, 0:2], B_RR is B[2:4, 2:4].
    # C is 2x4. C_L is C[:, 0:2], C_R is C[:, 2:4].
    
    # Block 0: A_L * B_LL
    # U indices: A_L (0,1). V indices: B_LL (0,1,4,5). W indices: C_L (0,1).
    # Mapping:
    # U_strassen[0] -> A11 -> A[0,0] (U[0])
    # U_strassen[1] -> A12 -> A[0,1] (U[1])
    # U_strassen[2] -> A21 -> A[1,0] (U[4])
    # U_strassen[3] -> A22 -> A[1,1] (U[5])
    
    # V_strassen[0] -> B11 -> B[0,0] (V[0])
    # V_strassen[1] -> B12 -> B[0,1] (V[1])
    # V_strassen[2] -> B21 -> B[1,0] (V[4])
    # V_strassen[3] -> B22 -> B[1,1] (V[5])
    
    # W_strassen[0] -> C11 -> C[0,0] (W[0])
    # W_strassen[1] -> C12 -> C[0,1] (W[1])
    # W_strassen[2] -> C21 -> C[1,0] (W[4])
    # W_strassen[3] -> C22 -> C[1,1] (W[5])
    
    def map_block(block_id, u_offset, v_offset, w_offset):
        # u_offset: index in U (0..7)
        # v_offset: index in V (0..15)
        # w_offset: index in W (0..7)
        for r in range(7):
            for i in range(4):
                if sU[i, r] != 0:
                    U[u_offset + i//2 * 2 + i%2, block_id*7 + r] += sU[i, r] # This mapping is wrong.
            
    # Let's do it manually for clarity and correctness.
    
    # Block 0: A_L (U[0:2, 0:2]) * B_LL (V[0:2, 0:2]) -> C_LL (W[0:2, 0:2])
    # U indices: 0,1 (A_L row 0), 4,5 (A_L row 1)
    # V indices: 0,1 (B_LL row 0), 4,5 (B_LL row 1)
    # W indices: 0,1 (C_LL row 0), 4,5 (C_LL row 1)
    
    # Block 1: A_R (U[0:2, 2:4]) * B_RL (V[2:4, 0:2]) -> C_LL (W[0:2, 0:2])
    # U indices: 2,3 (A_R row 0), 6,7 (A_R row 1)
    # V indices: 8,9 (B_RL row 0), 12,13 (B_RL row 1)
    # W indices: 0,1 (C_LL row 0), 4,5 (C_LL row 1)
    
    # Block 2: A_L (U[0:2, 0:2]) * B_LR (V[0:2, 2:4]) -> C_LR (W[0:2, 2:4])
    # U indices: 0,1, 4,5
    # V indices: 2,3 (B_LR row 0), 6,7 (B_LR row 1)
    # W indices: 2,3 (C_LR row 0), 6,7 (C_LR row 1)
    
    # Block 3: A_R (U[0:2, 2:4]) * B_RR (V[2:4, 2:4]) -> C_LR (W[0:2, 2:4])
    # U indices: 2,3, 6,7
    # V indices: 10,11 (B_RR row 0), 14,15 (B_RR row 1)
    # W indices: 2,3, 6,7
    
    # Helper to add a Strassen block
    def add_block(block_id, u_map, v_map, w_map):
        # u_map: list of 4 U indices corresponding to sU[0..3]
        # v_map: list of 4 V indices corresponding to sV[0..3]
        # w_map: list of 4 W indices corresponding to sW[0..3]
        for r in range(7):
            for i in range(4):
                val = sU[i, r]
                if val != 0:
                    U[u_map[i], block_id*7 + r] += val
                val = sV[i, r]
                if val != 0:
                    V[v_map[i], block_id*7 + r] += val
                val = sW[i, r]
                if val != 0:
                    W[w_map[i], block_id*7 + r] += val

    # Block 0
    add_block(0, [0,1,4,5], [0,1,4,5], [0,1,4,5])
    # Block 1
    add_block(1, [2,3,6,7], [8,9,12,13], [0,1,4,5])
    # Block 2
    add_block(2, [0,1,4,5], [2,3,6,7], [2,3,6,7])
    # Block 3
    add_block(3, [2,3,6,7], [10,11,14,15], [2,3,6,7])
    
    return U, V, W

def _m248_decomp():
    """
    Returns U, V, W for M_{2,4,8} (Rank 56).
    U: 8x56, V: 32x56, W: 16x56.
    """
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    u244, v244, w244 = _m244_decomp()
    
    # Block 0: A * B_L (B_L is 4x4, cols 0-3)
    # V indices 0-15. W indices 0-7.
    U[:, 0:28] = u244
    V[0:16, 0:28] = v244
    W[0:8, 0:28] = w244
    
    # Block 1: A * B_R (B_R is 4x4, cols 4-7)
    # V indices 16-31. W indices 8-15.
    U[:, 28:56] = u244
    V[16:32, 28:56] = v244
    W[8:16, 28:56] = w244
    
    return U, V, W

def _m148_decomp():
    """
    Returns U, V, W for M_{1,4,8} (Rank 32).
    U: 4x32, V: 32x32, W: 8x32.
    """
    rank = 32
    U = np.zeros((4, rank))
    V = np.zeros((32, rank))
    W = np.zeros((8, rank))
    
    for j in range(4):
        for k in range(8):
            r = j * 8 + k
            U[j, r] = 1
            V[j * 8 + k, r] = 1
            W[k, r] = 1
            
    return U, V, W

def _construct_88():
    """
    Construct rank 88 decomposition for M_{3,4,8}.
    """
    n, m, p = 3, 4, 8
    rank = 88
    U = np.zeros((n*m, rank)) # 12x88
    V = np.zeros((m*p, rank)) # 32x88
    W = np.zeros((p*n, rank)) # 24x88
    
    u248, v248, w248 = _m248_decomp()
    u148, v148, w148 = _m148_decomp()
    
    # Part 1: A_1 (rows 0,1) * B
    # U[0:2, 0:56] = u248
    # V[0:32, 0:56] = v248
    # W[0:16, 0:56] = w248 (C_1 is 2x8, indices 0..15 in W? No, W is p*n = 8*3 = 24)
    # W indices for C_1 (rows 0,1):
    # C[0, k] -> W[k*3 + 0]
    # C[1, k] -> W[k*3 + 1]
    # w248 is 16x56. w248[i, r] corresponds to C[i//4, i%4]?
    # No, w244 was 8x28. C is 2x4.
    # w248 is 16x56. C is 2x8.
    # w248[i, r] where i is 0..15.
    # i = row * 4 + col?
    # In _m244_decomp, W was 8x28.
    # W[0,0] = C[0,0]. W[1,0] = C[0,1]. W[2,0] = C[1,0]...
    # So w248[i, r] corresponds to C[i//4, i%4] for the first 8, and C[i//4, i%4+4] for next 8?
    # No, w248 was built from two w244s.
    # w248[0:8, 0:28] = w244 (C_L, 2x4)
    # w248[8:16, 28:56] = w244 (C_R, 2x4)
    # So w248[i, r] for i in 0..7 is C[i//4, i%4].
    # w248[i, r] for i in 8..15 is C[i//4 - 2, i%4 + 4].
    
    U[0:2, 0:56] = u248
    V[0:32, 0:56] = v248
    
    # Map W
    for i in range(16):
        if i < 8:
            c_row = i // 4
            c_col = i % 4
        else:
            c_row = i // 4 - 2
            c_col = i % 4 + 4
        w_idx = c_col * 3 + c_row
        W[w_idx, 0:56] = w248[i, :]
        
    # Part 2: A_2 (row 2) * B
    # U[2, 56:88] = u148
    # V[0:32, 56:88] = v148
    # W[?, 56:88] = w148
    # w148 is 8x32. C is 1x8.
    # w148[k, r] corresponds to C[0, k].
    # W index for C[2, k] is k*3 + 2.
    
    U[2, 56:88] = u148
    V[0:32, 56:88] = v148
    
    for k in range(8):
        w_idx = k * 3 + 2
        W[w_idx, 56:88] = w148[k, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Try rank 88
    try:
        decomp = _construct_88()
        T = _build_target_tensor(n, m, p)
        if _verify_decomposition(decomp["U"], decomp["V"], decomp["W"], T):
            return decomp
    except Exception:
        pass
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
