"""
Matrix multiplication 6x6x6 tensor decomposition.
Uses block Strassen on 3x3 blocks of 2x2 to achieve rank 189 (vs 216 schoolbook).
"""
import time
import numpy as np

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 6, 6, 6
    # Block size 2, so 3x3 blocks
    B = 2
    nb = n // B  # 3 blocks per dimension
    
    # Strassen 2x2 multiplication: 7 products
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U factors: for each Strassen step s, extract linear combo of A sub-block
    # A has shape (n, m) = (6,6). Block (ib, jb) is rows ib*B:(ib+1)*B, cols jb*B:(jb+1)*B
    # U[i*m+j, r] where i is row of A, j is col of A, r is rank index
    # For block (ib, jb) and step s, U picks up coefficients for the 2x2 sub-block
    
    # Strassen U coefficients (2x2): which entries of A-block are used and with what weight
    # M1: A11 + A22 -> [[1, 0], [0, 1]]
    # M2: A21 + A22 -> [[0, 0], [1, 1]]
    # M3: A11        -> [[1, 0], [0, 0]]
    # M4: A22        -> [[0, 0], [0, 1]]
    # M5: A11 + A12  -> [[1, 1], [0, 0]]
    # M6: A21 - A11  -> [[-1, 0], [1, 0]]
    # M7: A12 - A22  -> [[0, 1], [0, -1]]
    
    U_coeff = [
        np.array([[1, 0], [0, 1]]),   # M1
        np.array([[0, 0], [1, 1]]),   # M2
        np.array([[1, 0], [0, 0]]),   # M3
        np.array([[0, 0], [0, 1]]),   # M4
        np.array([[1, 1], [0, 0]]),   # M5
        np.array([[-1, 0], [1, 0]]),  # M6
        np.array([[0, 1], [0, -1]]),  # M7
    ]
    
    # V factors: which entries of B-block are used
    # M1: B11 + B22 -> [[1, 0], [0, 1]]
    # M2: B11       -> [[1, 0], [0, 0]]
    # M3: B12 - B22 -> [[0, 1], [0, -1]]
    # M4: B21 - B11 -> [[-1, 0], [1, 0]]
    # M5: B22       -> [[0, 0], [0, 1]]
    # M6: B11 + B12 -> [[1, 1], [0, 0]]
    # M7: B21 + B22 -> [[0, 1], [1, 1]]
    
    V_coeff = [
        np.array([[1, 0], [0, 1]]),   # M1
        np.array([[1, 0], [0, 0]]),   # M2
        np.array([[0, 1], [0, -1]]),  # M3
        np.array([[-1, 0], [1, 0]]),  # M4
        np.array([[0, 0], [0, 1]]),   # M5
        np.array([[1, 1], [0, 0]]),   # M6
        np.array([[0, 1], [1, 1]]),   # M7
    ]
    
    # W factors: reconstruction coefficients
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W[k*n+i, r]: row k (col of C), col i (row of C)
    # C block (ib, kb) is rows ib*B:(ib+1)*B, cols kb*B:(kb+1)*B
    # For step s, W gives contribution to C-sub-block from P_s
    
    # C11: [[1, 0], [0, 1]] for M1, [[0, 0], [0, 1]] for M4, [[-1, -1], [0, 0]] for M5, [[0, -1], [0, 1]] for M7
    # Actually let me be more careful.
    # C11 (top-left of C block) = M1_top_left + M4 - M5 + M7
    # M1 is a full 2x2, M4, M5, M7 are full 2x2s too.
    # C11 = M1[0,0] + M4 - M5 + M7  -- wait no.
    
    # Let me re-derive. In Strassen for 2x2:
    # C = [[C11, C12], [C21, C22]]
    # C11 = M1 + M4 - M5 + M7  (scalar combination of the 2x2 products? No...)
    # Actually each M_s is a scalar (product of two linear combos), and C entries are linear combos.
    # Wait, no. In standard Strassen, A and B are 2x2, each M_s is a scalar.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for each 2x2 block, the 4 entries of C are determined by the 7 scalars M_s.
    # When we embed into 6x6, each 2x2 block of A contributes to a 2x2 block of C.
    # The W factor for block (ib, kb) and step s tells us how P_s contributes to C[ib, kb].
    
    # W is a 2x2 matrix (for the 2x2 C block) for each step s.
    # C11 (entry [0,0] of block) gets: M1: 1, M4: 1, M5: -1, M7: 1, others: 0
    # C12 (entry [0,1] of block) gets: M3: 1, M5: 1, others: 0
    # C21 (entry [1,0] of block) gets: M2: 1, M4: 1, others: 0
    # C22 (entry [1,1] of block) gets: M1: 1, M2: -1, M3: 1, M6: 1, others: 0
    
    W_coeff = [
        np.array([[1, 0], [0, 1]]),   # M1: C11 += M1, C22 += M1
        np.array([[0, 0], [1, -1]]),  # M2: C21 += M2, C22 -= M2
        np.array([[0, 1], [0, 1]]),   # M3: C12 += M3, C22 += M3
        np.array([[1, 0], [1, 0]]),   # M4: C11 += M4, C21 += M4
        np.array([[-1, 1], [0, 0]]),  # M5: C11 -= M5, C12 += M5
        np.array([[0, 0], [0, 1]]),   # M6: C22 += M6
        np.array([[1, 0], [0, 0]]),   # M7: C11 += M7
    ]
    
    rank = nb * nb * nb * 7  # 3*3*3*7 = 189
    
    # Build U, V, W as dense matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for ib in range(nb):
        for jb in range(nb):
            for kb in range(nb):
                for s in range(7):
                    # U: block (ib, jb), step s
                    # Rows ib*B:(ib+1)*B, cols jb*B:(jb+1)*B of A
                    for di in range(B):
                        for dj in range(B):
                            i = ib * B + di
                            j = jb * B + dj
                            U[i * m + j, col] = U_coeff[s][di, dj]
                    
                    # V: block (jb, kb), step s
                    # Rows jb*B:(jb+1)*B, cols kb*B:(kb+1)*B of B
                    for dj in range(B):
                        for dk in range(B):
                            j = jb * B + dj
                            k = kb * B + dk
                            V[j * p + k, col] = V_coeff[s][dj, dk]
                    
                    # W: block (ib, kb), step s
                    # Rows kb*B:(kb+1)*B (k index), cols ib*B:(ib+1)*B (i index) of C
                    for dk in range(B):
                        for di in range(B):
                            k = kb * B + dk
                            i = ib * B + di
                            # W row index: k*n + i
                            W[k * n + i, col] = W_coeff[s][dk, di]
                    
                    col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    # Convert to list-of-lists format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
