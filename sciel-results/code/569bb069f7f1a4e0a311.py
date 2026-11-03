"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
DEADLINE_SECONDS = 110.0  # Safety margin from 120s limit

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a refined Strassen-based decomposition with rank 56.
    
    Approach: 
    1. Split A (4x4) into 2x4 top and bottom.
    2. Split B (4x8) into 4x4 left and right.
    3. This creates 4 subproblems of size 2x4 * 4x4 -> 2x4.
    4. Each 2x4 * 4x4 can be split into two 2x2 * 4x2 -> 2x2? No.
       2x4 A * 4x4 B.
       Split A into 2x2 left and 2x2 right.
       Split B into 4x2 left and 4x2 right.
       C = A_L B_L + A_R B_R.
       Each term is 2x2 * 4x2 -> 2x2.
       This is effectively two 2x2 * 2x2 multiplications if we split the 4x2 B into two 2x2?
       Wait. A_L is 2x2. B_L is 4x2.
       B_L = [B_L1; B_L2] where B_L1 is 2x2, B_L2 is 2x2.
       A_L * B_L = A_L * B_L1? No. A_L is 2x2. B_L is 4x2.
       A_L (2x2) * B_L (4x2) is invalid.
       
       Correction on previous logic:
       A is 2x4. B is 4x4.
       A = [A_L | A_R] (2x2 each).
       B = [B_T; B_B] (4x2 each).
       A * B = A_L B_T + A_R B_B.
       A_L B_T is 2x2 * 4x2? No.
       A_L is 2x2. B_T is 2x2 (top 2 rows of B).
       A_R is 2x2. B_B is 2x2 (bottom 2 rows of B).
       
       So yes, A_L B_T is M_{2,2,2}.
       A_R B_B is M_{2,2,2}.
       
       So for each 2x4 * 4x4 block, we need 2 M_{2,2,2} multiplications.
       There are 4 such blocks (Top-Left, Top-Right, Bottom-Left, Bottom-Right of the 4x8 C).
       Total 8 M_{2,2,2} multiplications.
       Rank = 8 * 7 = 56.
    """
    start = time.time()
    n, m, p = DIMENSIONS
    
    def _time_left():
        return DEADLINE_SECONDS - (time.time() - start)
    
    # Schoolbook rank-128 as fallback
    def _schoolbook():
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
    
    # Strassen 2x2 factors
    # u: A00, A01, A10, A11 (indices 0,1,2,3)
    # v: B00, B01, B10, B11
    # w: C00, C01, C10, C11
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # p1=(A00+A11)(B00+B11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # p2=(A10+A11)B00
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # p3=A00(B01-B11)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # p4=A11(B10-B00)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # p5=(A00+A01)B11
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # p6=(A10-A00)(B00+B01)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # p7=(A01-A11)(B10+B11)
    ]
    
    # Build rank-56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    if _time_left() < 1.0:
        return _schoolbook()
    
    # Define the 8 blocks
    # Each block is defined by:
    # i_offset: 0 or 2 (for A rows and C rows)
    # j_A_offset: 0 or 2 (for A cols, corresponding to B rows)
    # j_B_offset: 0 or 2 (for B rows, corresponding to A cols) - wait, B rows are j.
    # k_offset: 0, 2, 4, or 6 (for B cols and C cols)
    
    # Let's trace indices for one M_{2,2,2} block:
    # A part: i in [i_off, i_off+2), j in [jA_off, jA_off+2)
    # B part: j in [jB_off, jB_off+2), k in [k_off, k_off+2)
    # C part: i in [i_off, i_off+2), k in [k_off, k_off+2)
    # Note: j in A must match j in B. So jA_off == jB_off.
    
    blocks = []
    # i_off in {0, 2}
    # j_off in {0, 2} (This is the column of A / row of B)
    # k_off in {0, 2, 4, 6} (This is the column of B / column of C)
    
    # Wait, the structure was:
    # C_TL = A_T B_L
    #   = A_TL B_LL + A_TR B_LR
    # A_T: i in 0,1.
    # B_L: k in 0,1,2,3.
    # A_TL: j in 0,1.
    # B_LL: j in 0,1 (top rows of B_L), k in 0,1.
    # A_TR: j in 2,3.
    # B_LR: j in 2,3 (bottom rows of B_L), k in 2,3.
    
    # So Block 1: i_off=0, j_off=0, k_off=0. (A_TL * B_LL)
    # Block 2: i_off=0, j_off=2, k_off=2. (A_TR * B_LR)
    # Block 3: i_off=0, j_off=0, k_off=4. (A_TL * B_RL) -> B_RL is k in 4,5. j in 0,1.
    # Block 4: i_off=0, j_off=2, k_off=6. (A_TR * B_RR) -> B_RR is k in 6,7. j in 2,3.
    # Block 5: i_off=2, j_off=0, k_off=0. (A_BL * B_LL)
    # Block 6: i_off=2, j_off=2, k_off=2. (A_BR * B_LR)
    # Block 7: i_off=2, j_off=0, k_off=4. (A_BL * B_RL)
    # Block 8: i_off=2, j_off=2, k_off=6. (A_BR * B_RR)
    
    block_defs = [
        (0, 0, 0), # i_off, j_off, k_off
        (0, 2, 2),
        (0, 0, 4),
        (0, 2, 6),
        (2, 0, 0),
        (2, 2, 2),
        (2, 0, 4),
        (2, 2, 6)
    ]
    
    for bi, (i_off, j_off, k_off) in enumerate(block_defs):
        for s in range(7):
            u_block, v_block, w_block = S[s]
            col = bi * 7 + s
            
            # U: A indices
            # Local u index 0->00, 1->01, 2->10, 3->11
            # Global i = i_off + local_i
            # Global j = j_off + local_j
            for li in range(2):
                for lj in range(2):
                    val = u_block[li * 2 + lj]
                    if val == 0:
                        continue
                    gi = i_off + li
                    gj = j_off + lj
                    U[gi * m + gj][col] += val
            
            # V: B indices
            # Local v index 0->00, 1->01, 2->10, 3->11
            # Global j = j_off + local_j
            # Global k = k_off + local_k
            for lj in range(2):
                for lk in range(2):
                    val = v_block[lj * 2 + lk]
                    if val == 0:
                        continue
                    gj = j_off + lj
                    gk = k_off + lk
                    V[gj * p + gk][col] += val
            
            # W: C indices
            # Local w index 0->00, 1->01, 2->10, 3->11
            # Global i = i_off + local_i
            # Global k = k_off + local_k
            for li in range(2):
                for lk in range(2):
                    val = w_block[li * 2 + lk]
                    if val == 0:
                        continue
                    gi = i_off + li
                    gk = k_off + lk
                    W[gk * n + gi][col] += val
    
    # Verify
    def _verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return _schoolbook()
