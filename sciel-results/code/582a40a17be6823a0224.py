import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Strassen coefficients for 2x2 matrix multiplication
    # Each block maps 4-element indices for U, V, W
    # The 2x2 sub-blocks are indexed as:
    #   (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
    # So index 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    strassen_blocks = [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]}
    ]

    # We have 8 independent 2x2x2 sub-problems:
    # A is 2x4, B is 4x8, C is 2x8
    # Split A into two 2x2 blocks along columns: A[:, 0:2] and A[:, 2:4]
    # Split B into four 2x2 blocks along rows AND columns: B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4]
    # But actually for matrix mult C = A*B, we split differently:
    # A: 2 rows, 4 cols -> split into 2 blocks of 2 cols each: A_left (2x2), A_right (2x2)
    # B: 4 rows, 8 cols -> split into 4 blocks of 2 rows each: B_0, B_1, B_2, B_3 each 2x8
    #    Then each B_i split into 4 blocks of 2 cols: B_i_0, B_i_1, B_i_2, B_i_3 each 2x2
    # C: 2 rows, 8 cols -> split into 4 blocks of 2 cols each: C_0, C_1, C_2, C_3 each 2x2
    #
    # C_j = A_left * B_j_left + A_right * B_j_right where B_j = [B_j_left; B_j_right] (2x4)
    # So C_j = sum over the two A blocks times corresponding 2x4 B blocks
    # Each 2x4 B block = two 2x2 blocks side by side
    # So C_j (2x2) = A_left * B_{j,0} (2x2) + A_right * B_{j,1} (2x2)
    #
    # Total: 4 C-blocks, each = 2 Strassen multiplications = 8 Strassen = 8*7 = 56
    #
    # More carefully:
    # A is 2x4. Index A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x8. Index B[j, k] where j in {0,...,3}, k in {0,...,7}
    # C is 2x8. Index C[i, k] where i in {0,1}, k in {0,...,7}
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split A into A_L (cols 0,1) and A_R (cols 2,3), each 2x2
    # Split B into 4 row-pairs: B_0 (rows 0,1), B_1 (rows 2,3), but B is 4x8 so each is 2x8
    # Actually split B into 8 row-col 2x2 blocks: B_{r,c} where r in {0,1}, c in {0,1,2,3}
    # B_{r,c} uses rows {2r, 2r+1} and cols {2c, 2c+1}
    #
    # C is 2x8, split into 4 col-pairs: C_c (cols 2c, 2c+1), each 2x2
    #
    # C_c[i, k_local] = sum_{r, j_local} A_{r,j_local}[i, j_local] * B_{r,c}[j_local, k_local]
    #   where j_local in {0,1} corresponds to j = 2r + j_local
    #
    # So C_c = A_L * B_{0,c} + A_R * B_{1,c}
    # where A_L = A[:, 0:2] (2x2), A_R = A[:, 2:4] (2x2)
    # B_{r,c} = B[2r:2r+2, 2c:2c+2] (2x2)
    #
    # For each c in {0,1,2,3}: 2 Strassen multiplications
    # Total: 4 * 2 * 7 = 56
    #
    # Now, the factor matrices:
    # U has shape (n*m, rank) = (8, rank). Index: U[i*m + j, col] = U[i*4+j, col]
    # V has shape (m*p, rank) = (32, rank). Index: V[j*p + k, col] = V[j*8+k, col]
    # W has shape (p*n, rank) = (16, rank). Index: W[k*n + i, col] = W[k*2+i, col]
    #
    # For each (r, c) pair where r in {0,1}, c in {0,1,2,3}:
    #   This is one 2x2x2 Strassen multiplication
    #   A-block: A_{r} = A[:, 2r:2r+2]
    #     A_{r}[i, j_local] = A[i, 2r + j_local] = U[i*4 + 2r + j_local, col]
    #     i in {0,1}, j_local in {0,1}
    #     So a_ids = [0*4+2r+0, 0*4+2r+1, 1*4+2r+0, 1*4+2r+1] = [2r, 2r+1, 4+2r, 4+2r+1]
    #   B-block: B_{r,c} = B[2r:2r+2, 2c:2c+2]
    #     B_{r,c}[j_local, k_local] = B[2r+j_local, 2c+k_local] = V[(2r+j_local)*8 + 2c+k_local, col]
    #     j_local in {0,1}, k_local in {0,1}
    #     So b_ids = [(2r+0)*8+2c+0, (2r+0)*8+2c+1, (2r+1)*8+2c+0, (2r+1)*8+2c+1]
    #   C-block: C_c[i, k_local] where k = 2c + k_local
    #     C_c[i, k_local] = C[i, 2c+k_local] = W[(2c+k_local)*2 + i, col]
    #     i in {0,1}, k_local in {0,1}
    #     So c_ids = [(2c+0)*2+0, (2c+1)*2+0, (2c+0)*2+1, (2c+1)*2+1]
    #       = [4c, 4c+2, 4c+1, 4c+3]
    #     Wait: W[k*n+i] = W[(2c+k_local)*2 + i]
    #     For (i=0,k_local=0): W[0,0] -> W[0*2+0] = W[0] = index 0
    #     For (i=1,k_local=0): W[0,1] -> W[0*2+1] = W[1] = index 1
    #     For (i=0,k_local=1): W[1,0] -> W[1*2+0] = W[2] = index 2
    #     For (i=1,k_local=1): W[1,1] -> W[1*2+1] = W[3] = index 3
    #     So c_ids for the 4 positions (i,j_local) in the 2x2 C-block:
    #       position 0 (i=0,j_local=0): W[(2c+0)*2+0] = 4c
    #       position 1 (i=0,j_local=1): W[(2c+1)*2+0] = 4c+2
    #       position 2 (i=1,j_local=0): W[(2c+0)*2+1] = 4c+1
    #       position 3 (i=1,j_local=1): W[(2c+1)*2+1] = 4c+3
    #
    #     But in the Strassen pattern, the indices 0,1,2,3 correspond to:
    #       0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
    #     So c_ids = [4c, 4c+2, 4c+1, 4c+3]
    #
    # Let me re-derive:
    # For the 2x2 block, index mapping:
    #   local index 0 -> (row 0, col 0)
    #   local index 1 -> (row 0, col 1)
    #   local index 2 -> (row 1, col 0)
    #   local index 3 -> (row 1, col 1)
    #
    # U: A[i, j] where i is row, j is col of the 2x2 A-block
    #   local index = i*2 + j
    #   U index = i*m + (2r + j) = i*4 + 2r + j
    #   For local idx 0: i=0,j=0 -> 0*4+2r+0 = 2r
    #   For local idx 1: i=0,j=1 -> 0*4+2r+1 = 2r+1
    #   For local idx 2: i=1,j=0 -> 1*4+2r+0 = 4+2r
    #   For local idx 3: i=1,j=1 -> 1*4+2r+1 = 5+2r
    #   a_ids = [2r, 2r+1, 4+2r, 5+2r]
    #
    # V: B[j, k] where j is row, k is col of the 2x2 B-block
    #   local index = j*2 + k
    #   V index = (2r+j)*p + (2c+k) = (2r+j)*8 + 2c + k
    #   For local idx 0: j=0,k=0 -> 2r*8 + 2c
    #   For local idx 1: j=0,k=1 -> 2r*8 + 2c + 1
    #   For local idx 2: j=1,k=0 -> (2r+1)*8 + 2c
    #   For local idx 3: j=1,k=1 -> (2r+1)*8 + 2c + 1
    #   b_ids = [2r*8+2c, 2r*8+2c+1, (2r+1)*8+2c, (2r+1)*8+2c+1]
    #
    # W: C[i, k] where i is row, k is col of the 2x2 C-block
    #   local index = i*2 + k
    #   W index = (2c+k)*n + i = (2c+k)*2 + i
    #   For local idx 0: i=0,k=0 -> 4c
    #   For local idx 1: i=0,k=1 -> 4c+2
    #   For local idx 2: i=1,k=0 -> 4c+1
    #   For local idx 3: i=1,k=1 -> 4c+3
    #   c_ids = [4c, 4c+2, 4c+1, 4c+3]

    rank = 56
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)

    for r in range(2):
        for c in range(4):
            a_ids = [2*r, 2*r+1, 4+2*r, 5+2*r]
            b_ids = [2*r*8+2*c, 2*r*8+2*c+1, (2*r+1)*8+2*c, (2*r+1)*8+2*c+1]
            c_ids = [4*c, 4*c+2, 4*c+1, 4*c+3]

            base_col = 7 * (4 * r + c)

            for s_idx, pattern in enumerate(strassen_blocks):
                col = base_col + s_idx

                for idx, coeff in pattern['u']:
                    U[a_ids[idx], col] += coeff
                for idx, coeff in pattern['v']:
                    V[b_ids[idx], col] += coeff
                for idx, coeff in pattern['w']:
                    W[c_ids[idx], col] += coeff

    # Verify base solution
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
