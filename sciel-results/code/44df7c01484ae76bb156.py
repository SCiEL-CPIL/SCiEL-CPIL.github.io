"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
TIME_LIMIT = 145  # seconds, safety margin from 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    start_time = time.time()
    
    # Build schoolbook decomposition as baseline
    rank_schoolbook = n * m * p  # 128
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Verify schoolbook works
    def verify(U, V, W, tol=1e-10):
        """Check that sum_r U[:,r] * V[:,r] * W[:,r] == multiplication tensor"""
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # Tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # Expected: T[i,j,k] = 1 if this is the identity for matrix mult
        # Actually the tensor for matrix mult A(n,m) x B(m,p) -> C(n,p)
        # T[i,j,k] should be delta (the multiplication tensor)
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # Wait no. The multiplication tensor T has T[i,j,k] = 1
        # because C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=tol)
    
    # Try to use Strassen-like decomposition to reduce rank
    # For 4x4 x 4x8, we can split B into two 4x4 matrices
    # Use Strassen for each 4x4 x 4x4 block
    # This gives rank = 2 * 49 = 98 (using Strassen's 7 multiplications for 2x2x2 blocks)
    # Actually Strassen for 2x2x2 has rank 7, so for 4x4x4 it's 7*4 = 28
    # Wait, Strassen recursively: 4x4x4 = 7 * (2x2x2) = 7 * 7 = 49
    # For 4x4 x 4x8 = 2 * (4x4 x 4x4), so rank = 2 * 49 = 98
    
    def strassen_decompose_4x4x4(n=4, m=4, p=4):
        """Build Strassen decomposition for 4x4 x 4x4 -> 4x4.
        Returns U (16, rank), V (16, rank), W (16, rank)"""
        # Use 2 levels of Strassen: 4x4x4 = 7 * (2x2x2) = 49 * (1x1x1)
        # Level 1: split 4x4 into 2x2 blocks
        # A = [[A00, A01], [A10, A11]] where each is 2x2
        # B = [[B00, B01], [B10, B11]] where each is 2x2
        # C = [[C00, C01], [C10, C11]] where each is 2x2
        
        # Strassen multipliers (each is a 2x2x2 problem):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Each Mi is a 2x2x2 multiplication, which we decompose with Strassen (rank 7)
        
        # Build the full decomposition
        U = []
        V = []
        W = []
        
        # For each of the 7 Strassen multipliers at level 1,
        # we have a 2x2x2 sub-problem with rank 7
        # Total rank = 7 * 7 = 49
        
        # Define index helpers
        def A_idx_4x4(block_r, block_c, local_r, local_c):
            """Index into U for A[2*block_r+local_r, 2*block_c+local_c]"""
            i = 2 * block_r + local_r
            j = 2 * block_c + local_c
            return i * 4 + j
        
        def B_idx_4x4(block_r, block_c, local_r, local_c):
            """Index into V for B[2*block_r+local_r, 2*block_c+local_c]"""
            j = 2 * block_r + local_r
            k = 2 * block_c + local_c
            return j * 4 + k
        
        def C_idx_4x4(block_r, block_c, local_r, local_c):
            """Index into W for C[2*block_r+local_r, 2*block_c+local_c]"""
            i = 2 * block_r + local_r
            k = 2 * block_c + local_c
            return k * 4 + i
        
        # Strassen multipliers for 2x2x2 (rank 7)
        # For X (2x2), Y (2x2) -> Z (2x2):
        # s1 = (x00+x11)(y00+y11)
        # s2 = (x10+x11)(y00)
        # s3 = (x00)(y01-y11)
        # s4 = (x11)(y10-y00)
        # s5 = (x00+x01)(y11)
        # s6 = (x10-x00)(y00+y01)
        # s7 = (x01-x11)(y10+y11)
        # z00 = s1 + s4 - s5 - s7
        # z01 = s3 + s5
        # z10 = s2 + s4
        # z11 = s1 - s2 + s3 + s7
        
        # For each level-1 multiplier, define the X and Y blocks
        # and their coefficients
        
        # Level 1 multipliers:
        # M1: X = A00+A11, Y = B00+B11
        # M2: X = A10+A11, Y = B00
        # M3: X = A00, Y = B01-B11
        # M4: X = A11, Y = B10-B00
        # M5: X = A00+A01, Y = B11
        # M6: X = A10-A00, Y = B00+B01
        # M7: X = A01-A11, Y = B10+B11
        
        # For each, X is a linear combination of 2x2 A blocks,
        # Y is a linear combination of 2x2 B blocks
        
        # Define the level-1 multipliers as (X_coeff, Y_coeff)
        # X_coeff: dict from (block_r, block_c) to coefficient
        # Y_coeff: dict from (block_r, block_c) to coefficient
        
        level1_mults = [
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, 0),  # contributes to C00, C11
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, 1),  # contributes to C10, C11
            # M3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, 2),  # contributes to C01, C11
            # M4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, 3),  # contributes to C00, C10
            # M5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, 4),  # contributes to C00, C01
            # M6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, 5),  # contributes to C11
            # M7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, 6),  # contributes to C00, C11
        ]
        
        # Output contributions: which C blocks each M contributes to, with signs
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        output_contribs = [
            # For each level-1 multiplier, dict from (block_r, block_c) to coefficient
            {(0,0): 1, (1,1): 1},   # M1 -> C00, C11
            {(1,0): 1, (1,1): -1},  # M2 -> C10, C11
            {(0,1): 1, (1,1): 1},   # M3 -> C01, C11
            {(0,0): 1, (1,0): 1},   # M4 -> C00, C10
            {(0,0): -1, (0,1): 1},  # M5 -> C00, C01
            {(1,1): 1},             # M6 -> C11
            {(0,0): -1, (1,1): 1},  # M7 -> C00, C11
        ]
        
        # Now build the full 4x4x4 decomposition
        # For each level-1 multiplier m (0-6), and each Strassen sub-multiplier s (0-6),
        # we get one rank-1 term
        
        rank = 49
        U_mat = np.zeros((16, rank), dtype=np.float64)
        V_mat = np.zeros((16, rank), dtype=np.float64)
        W_mat = np.zeros((16, rank), dtype=np.float64)
        
        col = 0
        for m_idx in range(7):
            X_coeff, Y_coeff, _ = level1_mults[m_idx]
            out_contrib = output_contribs[m_idx]
            
            for s_idx in range(7):
                # Determine which elements of X and Y are involved in sub-multiplier s
                # and their coefficients
                # Also determine which elements of Z are produced
                
                if s_idx == 0:  # s1 = (x00+x11)(y00+y11)
                    x_elems = {((0,0), 0, 0): 1, ((1,1), 0, 0): 1}
                    y_elems = {((0,0), 0, 0): 1, ((1,1), 0, 0): 1}
                    z_elems = {((0,0), 0, 0): 1, ((1,1), 0, 0): 1}
                elif s_idx == 1:  # s2 = (x10+x11)(y00)
                    x_elems = {((1,0), 0, 0): 1, ((1,1), 0, 0): 1}
                    y_elems = {((0,0), 0, 0): 1}
                    z_elems = {((1,0), 0, 0): 1, ((1,1), 0, 0): 1}
                elif s_idx == 2:  # s3 = (x00)(y01-y11)
                    x_elems = {((0,0), 0, 0): 1}
                    y_elems = {((0,1), 0, 0): 1, ((1,1), 0, 0): -1}
                    z_elems = {((0,1), 0, 0): 1, ((1,1), 0, 0): 1}
                elif s_idx == 3:  # s4 = (x11)(y10-y00)
                    x_elems = {((1,1), 0, 0): 1}
                    y_elems = {((1,0), 0, 0): 1, ((0,0), 0, 0): -1}
                    z_elems = {((0,0), 0, 0): 1, ((1,0), 0, 0): 1}
                elif s_idx == 4:  # s5 = (x00+x01)(y11)
                    x_elems = {((0,0), 0, 0): 1, ((0,1), 0, 0): 1}
                    y_elems = {((1,1), 0, 0): 1}
                    z_elems = {((0,0), 0, 0): -1, ((0,1), 0, 0): 1}
                elif s_idx == 5:  # s6 = (x10-x00)(y00+y01)
                    x_elems = {((1,0), 0, 0): 1, ((0,0), 0, 0): -1}
                    y_elems = {((0,0), 0, 0): 1, ((0,1), 0, 0): 1}
                    z_elems = {((1,1), 0, 0): 1}
                elif s_idx == 6:  # s7 = (x01-x11)(y10+y11)
                    x_elems = {((0,1), 0, 0): 1, ((1,1), 0, 0): -1}
                    y_elems = {((1,0), 0, 0): 1, ((1,1), 0, 0): 1}
                    z_elems = {((0,0), 0, 0): -1, ((1,1), 0, 0): 1}
                else:
                    raise ValueError("Invalid s_idx")
                
                # For each element in x_elems, combine with X_coeff
                for (x_block, x_lr, x_lc), x_c in x_elems.items():
                    for (xb, xc), xbc in X_coeff.items():
                        # The global A index
                        ai = A_idx_4x4(xb, xc, x_lr, x_lc)
                        U_mat[ai, col] += x_c * xbc
                
                for (y_block, y_lr, y_lc), y_c in y_elems.items():
                    for (yb, yc), ybc in Y_coeff.items():
                        # The global B index
                        bj = B_idx_4x4(yb, yc, y_lr, y_lc)
                        V_mat[bj, col] += y_c * ybc
                
                for (z_block, z_lr, z_lc), z_c in z_elems.items():
                    for (zb, zc), zbc in out_contrib.items():
                        # The global C index
                        ci = C_idx_4x4(zb, zc, z_lr, z_lc)
                        W_mat[ci, col] += z_c * zbc
                
                col += 1
        
        return U_mat, V_mat, W_mat
    
    # Build the 4x4 x 4x8 decomposition using two 4x4x4 Strassen decompositions
    # B is split into B_left (4x4) and B_right (4x4)
    # C is split into C_left (4x4) and C_right (4x4)
    
    U_left, V_left, W_left = strassen_decompose_4x4x4()
    U_right, V_right, W_right = strassen_decompose_4x4x4()
    
    # Combine: U is the same for both (since A is the same)
    # V_left uses B[0:4, 0:4], V_right uses B[0:4, 4:8]
    # W_left uses C[0:4, 0:4], W_right uses C[0:4, 4:8]
    
    rank_strassen = 98
    U_full = np.zeros((16, rank_strassen), dtype=np.float64)
    V_full = np.zeros((32, rank_strassen), dtype=np.float64)
    W_full = np.zeros((32, rank_strassen), dtype=np.float64)
    
    # Left half (B cols 0-3, C cols 0-3)
    U_full[:, 0:49] = U_left
    V_full[0:16, 0:49] = V_left  # B rows 0-3, cols 0-3 -> V index j*8+k where j in [0,4), k in [0,4)
    # Wait, V has shape (m*p, rank) = (32, rank). V[j*p+k, r] = V[j*8+k, r]
    # For B_left, j in [0,4), k in [0,4), so V index = j*8+k
    # V_left has shape (16, 49) with index j*4+k for j,k in [0,4)
    # We need to map V_left[j*4+k] -> V_full[j*8+k]
    for j in range(4):
        for k in range(4):
            V_full[j * 8 + k, 0:49] = V_left[j * 4 + k, 0:49]
    
    # W_left has shape (16, 49) with index k*4+i for k,i in [0,4)
    # W_full has shape (32, 98) with index k*4+i for k in [0,8), i in [0,4)
    # For C_left, k in [0,4), so W index = k*4+i
    W_full[0:16, 0:49] = W_left
    
    # Right half (B cols 4-7, C cols 4-7)
    U_full[:, 49:98] = U_right
    # For B_right, j in [0,4), k in [4,8), so V index = j*8+k
    # V_right has shape (16, 49) with index j*4+k for j,k in [0,4)
    # We need to map V_right[j*4+k] -> V_full[j*8+(k+4)]
    for j in range(4):
        for k in range(4):
            V_full[j * 8 + (k + 4), 49:98] = V_right[j * 4 + k, 0:49]
    
    # For C_right, k in [4,8), so W index = k*4+i = (k-4+4)*4+i
    # W_right has shape (16, 49) with index k*4+i for k,i in [0,4)
    # We need to map W_right[k*4+i] -> W_full[(k+4)*4+i]
    for k in range(4):
        for i in range(4):
            W_full[(k + 4) * 4 + i, 49:98] = W_right[k * 4 + i, 0:49]
    
    # Verify
    if verify(U_full, V_full, W_full):
        result = {
            "U": to_list(U_full),
            "V": to_list(V_full),
            "W": to_list(W_full)
        }
        return result
    
    # If verification fails, fall back to schoolbook
    return {
        "U": to_list(U_sb),
        "V": to_list(V_sb),
        "W": to_list(W_sb)
    }
# EVOLVE-BLOCK-END
