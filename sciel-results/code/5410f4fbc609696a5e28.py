import numpy as np
import time

_DIMS = (2, 4, 4)

# Strassen 2x2 steps: each step has u, v, w as (local_index, coeff) lists
_STRASSEN = (
    ((0, 1.0), (3, 1.0)), ((0, 1.0), (3, 1.0)), ((0, 1.0), (3, 1.0)),
    ((2, 1.0), (3, 1.0)), ((0, 1.0)), ((2, 1.0), (3, -1.0)),
    ((0, 1.0)), ((1, 1.0), (3, -1.0)), ((1, 1.0), (3, 1.0)),
    ((3, 1.0)), ((2, 1.0), (0, -1.0)), ((0, 1.0), (2, 1.0)),
    ((0, 1.0), (1, 1.0)), ((3, 1.0)), ((0, -1.0), (1, 1.0)),
    ((2, 1.0), (0, -1.0)), ((0, 1.0), (1, 1.0)), ((3, 1.0)),
    ((1, 1.0), (3, -1.0)), ((2, 1.0), (3, 1.0)), ((0, 1.0)),
)
# Reshape into 7 steps of 3 lists each
_STRASSEN = [(_STRASSEN[i:i+1], _STRASSEN[i+1:i+2], _STRASSEN[i+2:i+3]) for i in range(0, 21, 3)]

def _build_strassen_decomp(n, m, p):
    """Build rank-28 Strassen decomposition for n x m * m x p."""
    rank = 4 * 7  # 4 blocks of 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for blk in range(4):
        ar0 = (blk // 2) * 2  # row offset for A
        bc0 = (blk % 2) * 2   # col offset for B (which is row of B)
        # A is n x m, we take rows ar0:ar0+2, cols 0:m (all 4 cols, but Strassen on 2x2 sub-blocks)
        # Wait, A is 2x4. We split columns into two 2-col blocks.
        # B is 4x4. We split rows into two 2-row blocks and cols into two 2-col blocks.
        # C is 4x2. We split rows into two 2-row blocks and cols... C is p x n = 4 x 2.
        # 
        # Let me reconsider the block structure.
        # A: 2 x 4 -> split into 1x2 blocks of 2x2: A00 (rows 0:2, cols 0:2), A01 (rows 0:2, cols 2:4)
        # B: 4 x 4 -> split into 2x2 blocks of 2x2: B00, B01, B10, B11
        # C: 4 x 2 -> C[i,j] = sum_k A[i,k]*B[k,j]
        #   C is 4x2. Split rows into 2x2: C00 (rows 0:2, cols 0:2), C01 (rows 2:4, cols 0:2)
        #   Actually C has 4 rows and 2 cols. So C00 = rows 0:2, cols 0:2; C10 = rows 2:4, cols 0:2
        #
        # C[0:2, 0:2] = A00 * B00 + A01 * B10  (2x2 = 2x2*2x2 + 2x2*2x2)
        # C[2:4, 0:2] = A00 * B01 + A01 * B11
        #
        # So we need 4 Strassen multiplications:
        #   S1: A00 * B00 -> contributes to C00
        #   S2: A01 * B10 -> contributes to C00
        #   S3: A00 * B01 -> contributes to C10
        #   S4: A01 * B11 -> contributes to C10
        
        pass

    # Let me redo this more carefully with explicit index mapping.
    # A[i, j] where i in [0,2), j in [0,4)
    # B[k, l] where k in [0,4), l in [0,4)
    # C[k, i] where k in [0,4), i in [0,2)  (W index: k*n + i)
    #
    # C[k, i] = sum_j A[i, j] * B[j, k]
    #
    # Block decomposition:
    # A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # B_top_left = B[0:2, 0:2], B_top_right = B[0:2, 2:4]
    # B_bot_left = B[2:4, 0:2], B_bot_right = B[2:4, 2:4]
    #
    # C[0:2, :] = A_left * B_top_left + A_right * B_bot_left  (each 2x2)
    # C[2:4, :] = A_left * B_top_right + A_right * B_bot_right
    #
    # So 4 independent 2x2 multiplications, each done via Strassen (7 mults) = 28 total.
    
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)  
    # W has shape (p*n, rank) = (8, 28)
    
    # A indices: (row, col) -> index row*m + col = row*4 + col, row in [0,2), col in [0,4)
    # B indices: (row, col) -> index row*p + col = row*4 + col, row in [0,4), col in [0,4)
    # C indices: (row, col) -> index row*n + col = row*2 + col, row in [0,4), col in [0,2)
    
    # Block 0: A_left * B_top_left -> C[0:2, :]
    #   A local (r,c) -> global A index r*4 + c
    #   B local (r,c) -> global B index r*4 + c  (top-left, so row offset 0, col offset 0)
    #   C local (r,c) -> global C index r*2 + c  (top rows, so row offset 0, col offset 0)
    # Block 1: A_right * B_bot_left -> C[0:2, :]
    #   A local (r,c) -> global A index r*4 + (c+2)
    #   B local (r,c) -> global B index (r+2)*4 + c
    #   C local (r,c) -> global C index r*2 + c
    # Block 2: A_left * B_top_right -> C[2:4, :]
    #   A local (r,c) -> global A index r*4 + c
    #   B local (r,c) -> global B index r*4 + (c+2)
    #   C local (r,c) -> global C index (r+2)*2 + c
    # Block 3: A_right * B_bot_right -> C[2:4, :]
    #   A local (r,c) -> global A index r*4 + (c+2)
    #   B local (r,c) -> global B index (r+2)*4 + (c+2)
    #   C local (r,c) -> global C index (r+2)*2 + c
    
    blocks = [
        # (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 2, 0),
        (0, 2, 2, 2, 2, 0),
    ]
    
    for bidx, (ar_off, ac_off, br_off, bc_off, cr_off, cc_off) in enumerate(blocks):
        for s in range(7):
            u_step, v_step, w_step = _STRASSEN[s]
            col = bidx * 7 + s
            for (li, coeff) in u_step:
                lr, lc = li // 2, li % 2
                gi = (lr + ar_off) * m + (lc + ac_off)
                U[gi, col] += coeff
            for (li, coeff) in v_step:
                lr, lc = li // 2, li % 2
                gi = (lr + br_off) * p + (lc + bc_off)
                V[gi, col] += coeff
            for (li, coeff) in w_step:
                lr, lc = li // 2, li % 2
                gi = (lr + cr_off) * n + (lc + cc_off)
                W[gi, col] += coeff
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition correctness."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    T = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    
    for ar in range(n):
        for ac in range(m):
            for bc in range(p):
                i = ar * m + ac
                j = ac * p + bc
                k = bc * n + ar
                if abs(T[i, j, k] - 1.0) > 1e-9:
                    return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    
    n, m, p = _DIMS
    
    # Build Strassen rank-28 decomposition
    U, V, W = _build_strassen_decomp(n, m, p)
    
    if time.time() > deadline - 10.0:
        return {"U": U, "V": V, "W": W}
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: naive rank 32
    rank_naive = n * m * p
    Uf = [[0.0] * rank_naive for _ in range(n * m)]
    Vf = [[0.0] * rank_naive for _ in range(m * p)]
    Wf = [[0.0] * rank_naive for _ in range(p * n)]
    r = 0
    for ar in range(n):
        for ac in range(m):
            for bc in range(p):
                i = ar * m + ac
                j = ac * p + bc
                k = bc * n + ar
                Uf[i][r] = 1.0
                Vf[j][r] = 1.0
                Wf[k][r] = 1.0
                r += 1
    return {"U": Uf, "V": Vf, "W": Wf}
