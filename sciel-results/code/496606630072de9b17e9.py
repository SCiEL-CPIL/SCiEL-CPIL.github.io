"""Matrix multiplication 4x4x9 rank-minimal decomposition.

Strategy:
1. Construct an explicit Strassen-based rank-49 decomposition for 4x4*4x4.
2. Combine two 4x4*4x4 Strassen blocks (for columns 0-7) with schoolbook
   for the last column (column 8) to get rank 49+49+16 = 114.
3. Verify the decomposition is exact.
4. If time permits, try to reduce further via ALS refinement.
"""
import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 4, 4, 9
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36

    # Build the target tensor triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))

    triple_set = set(triples)
    num_triples = len(triples)  # 144

    # ------------------------------------------------------------------
    # Step 1: Build Strassen rank-7 decomposition for 2x2*2x2
    # ------------------------------------------------------------------
    # For 2x2 matrices:
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # C = A*B = [[c0, c1], [c2, c3]]
    #
    # Strassen's 7 products:
    # M1 = (a0+a3)*(b0+b3)
    # M2 = (a2+a3)*b0
    # M3 = a0*(b1-b3)
    # M4 = a3*(b2-b0)
    # M5 = (a0+a1)*b3
    # M6 = (a2-a0)*(b0+b1)
    # M7 = (a1-a3)*(b2+b3)
    #
    # C entries:
    # c0 = M1 + M4 - M5 + M7
    # c1 = M3 + M5
    # c2 = M2 + M4
    # c3 = M1 - M2 + M3 + M6

    def build_strassen_2x2():
        """Build rank-7 U,V,W for 2x2*2x2 matrix multiplication.
        
        Returns U (4x7), V (4x7), W (4x7) where:
        U[i*2+j, r] corresponds to A[i,j]
        V[i*2+j, r] corresponds to B[i,j]  
        W[i*2+j, r] corresponds to C[i,j]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a0+a3)*(b0+b3)
        U[0, 0] = 1; U[3, 0] = 1  # a0 + a3
        V[0, 0] = 1; V[3, 0] = 1  # b0 + b3
        W[0, 0] = 1; W[3, 0] = 1  # c0 += M1, c3 += M1
        
        # M2 = (a2+a3)*b0
        U[2, 1] = 1; U[3, 1] = 1  # a2 + a3
        V[0, 1] = 1  # b0
        W[2, 1] = 1  # c2 += M2
        W[3, 1] = -1  # c3 -= M2
        
        # M3 = a0*(b1-b3)
        U[0, 2] = 1  # a0
        V[1, 2] = 1; V[3, 2] = -1  # b1 - b3
        W[1, 2] = 1  # c1 += M3
        W[3, 2] = 1  # c3 += M3
        
        # M4 = a3*(b2-b0)
        U[3, 3] = 1  # a3
        V[2, 3] = 1; V[0, 3] = -1  # b2 - b0
        W[0, 3] = 1  # c0 += M4
        W[2, 3] = 1  # c2 += M4
        
        # M5 = (a0+a1)*b3
        U[0, 4] = 1; U[1, 4] = 1  # a0 + a1
        V[3, 4] = 1  # b3
        W[0, 4] = -1  # c0 -= M5
        W[1, 4] = 1  # c1 += M5
        
        # M6 = (a2-a0)*(b0+b1)
        U[2, 5] = 1; U[0, 5] = -1  # a2 - a0
        V[0, 5] = 1; V[1, 5] = 1  # b0 + b1
        W[3, 5] = 1  # c3 += M6
        
        # M7 = (a1-a3)*(b2+b3)
        U[1, 6] = 1; U[3, 6] = -1  # a1 - a3
        V[2, 6] = 1; V[3, 6] = 1  # b2 + b3
        W[0, 6] = 1  # c0 += M7
        
        return U, V, W

    # ------------------------------------------------------------------
    # Step 2: Build Strassen rank-49 decomposition for 4x4*4x4
    # ------------------------------------------------------------------
    def build_strassen_4x4():
        """Build rank-49 U,V,W for 4x4*4x4 matrix multiplication.
        
        Uses Strassen at the block level (2x2 blocks of 2x2), each block
        product using the rank-7 Strassen decomposition.
        
        U (16x49), V (16x49), W (16x49)
        """
        U2, V2, W2 = build_strassen_2x2()
        
        # 4x4 as 2x2 block matrix:
        # A = [[A11, A12], [A21, A22]] where each block is 2x2
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        #
        # Block-level Strassen:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # Block index mapping:
        # A11: rows 0-3, A12: rows 4-7, A21: rows 8-11, A22: rows 12-15
        # Same for B and C.
        
        # For each of the 7 block-level Strassen multipliers,
        # we need to construct U, V contributions and then
        # each produces a 2x2*2x2 product (rank 7).
        
        # Block-level U coefficients (which A blocks participate):
        # M1: A11(+1), A22(+1)
        # M2: A21(+1), A22(+1)
        # M3: A11(+1)
        # M4: A22(+1)
        # M5: A11(+1), A12(+1)
        # M6: A21(+1), A11(-1)
        # M7: A12(+1), A22(-1)
        
        block_U_coeffs = [
            {0: 1, 3: 1},   # M1: A11, A22
            {2: 1, 3: 1},   # M2: A21, A22
            {0: 1},          # M3: A11
            {3: 1},          # M4: A22
            {0: 1, 1: 1},   # M5: A11, A12
            {2: 1, 0: -1},  # M6: A21, -A11
            {1: 1, 3: -1},  # M7: A12, -A22
        ]
        
        # Block-level V coefficients (which B blocks participate):
        # M1: B11(+1), B22(+1)
        # M2: B11(+1)
        # M3: B12(+1), B22(-1)
        # M4: B21(+1), B11(-1)
        # M5: B22(+1)
        # M6: B11(+1), B12(+1)
        # M7: B21(+1), B22(+1)
        
        block_V_coeffs = [
            {0: 1, 3: 1},   # M1: B11, B22
            {0: 1},          # M2: B11
            {1: 1, 3: -1},  # M3: B12, -B22
            {2: 1, 0: -1},  # M4: B21, -B11
            {3: 1},          # M5: B22
            {0: 1, 1: 1},   # M6: B11, B12
            {2: 1, 3: 1},   # M7: B21, B22
        ]
        
        # Block-level W coefficients (which C blocks get contributions):
        # C11: M1(+1), M4(+1), M5(-1), M7(+1)
        # C12: M3(+1), M5(+1)
        # C21: M2(+1), M4(+1)
        # C22: M1(+1), M2(-1), M3(+1), M6(+1)
        
        block_W_coeffs = [
            {0: 1, 3: 1, 1: -1, 7: 1},  # C11
            {3: 1, 5: 1},                 # C12
            {2: 1, 4: 1},                 # C21
            {1: 1, 2: -1, 3: 1, 6: 1},   # C22
        ]
        
        # Actually, let me re-derive this more carefully.
        # The 7 block products M1..M7 each are 2x2*2x2 multiplications.
        # For each block product Mr, we use the rank-7 Strassen decomposition.
        # The U and V for Mr are formed by combining the block U/V coefficients
        # with the 2x2 Strassen U/V.
        
        # Let me restructure: for each of the 7 block multipliers,
        # and for each of the 7 sub-multiplications within that,
        # we get one rank-1 term.
        
        # Total: 7 * 7 = 49 rank-1 terms.
        
        # Block row offsets in the 4x4 matrix (each block is 2x2 = 4 entries):
        # Block 0 (A11/B11/C11): entries 0,1,2,3
        # Block 1 (A12/B12/C12): entries 4,5,6,7
        # Block 2 (A21/B21/C21): entries 8,9,10,11
        # Block 3 (A22/B22/C22): entries 12,13,14,15
        
        col = 0
        for block_mult in range(7):  # 7 block-level Strassen multipliers
            u_blocks = block_U_coeffs[block_mult]  # {block_idx: coeff}
            v_blocks = block_V_coeffs[block_mult]  # {block_idx: coeff}
            
            for sub_mult in range(7):  # 7 sub-level Strassen multipliers
                # U contribution: for each A block with nonzero coeff,
                # add coeff * U2[:, sub_mult] at the block's row offset
                for blk, coeff in u_blocks.items():
                    row_offset = blk * 4
                    U4[row_offset:row_offset+4, col] = coeff * U2[:, sub_mult]
                
                # V contribution
                for blk, coeff in v_blocks.items():
                    row_offset = blk * 4
                    V4[row_offset:row_offset+4, col] = coeff * V2[:, sub_mult]
                
                # W contribution: need to figure out which C blocks this
                # block multiplier contributes to and with what coefficient.
                # Then for each such C block, add coeff * W2[:, sub_mult]
                
                # Determine C block contributions from this block multiplier:
                # M1 -> C11(+1), C22(+1)
                # M2 -> C21(+1), C22(-1)
                # M3 -> C12(+1), C22(+1)
                # M4 -> C11(+1), C21(+1)
                # M5 -> C11(-1), C12(+1)
                # M6 -> C22(+1)
                # M7 -> C11(+1)
                
                c_block_map = {
                    0: {0: 1, 3: 1},    # M1 -> C11, C22
                    1: {2: 1, 3: -1},   # M2 -> C21, -C22
                    2: {1: 1, 3: 1},    # M3 -> C12, C22
                    3: {0: 1, 2: 1},    # M4 -> C11, C21
                    4: {0: -1, 1: 1},   # M5 -> -C11, C12
                    5: {3: 1},           # M6 -> C22
                    6: {0: 1},           # M7 -> C11
                }
                
                c_blocks = c_block_map[block_mult]
                for blk, coeff in c_blocks.items():
                    row_offset = blk * 4
                    W4[row_offset:row_offset+4, col] = coeff * W2[:, sub_mult]
                
                col += 1
        
        return U4, V4, W4

    # ------------------------------------------------------------------
    # Step 3: Build the full 4x4 * 4x9 decomposition
    # ------------------------------------------------------------------
    def build_full_decomposition():
        """Build the rank-114 decomposition:
        - 49 terms for columns 0-3 (first 4x4 block of B)
        - 49 terms for columns 4-7 (second 4x4 block of B)
        - 16 terms for column 8 (single column, schoolbook)
        Total: 114
        """
        U4, V4, W4 = build_strassen_4x4()
        
        rank = 114
        U = np.zeros((16, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        
        # Block 1: columns 0-3 of B (B11 in 4x4 terms)
        # V rows for this block: j*9 + k for j in 0..3, k in 0..3
        # V4 has 16 rows (4x4), we need to map to 36 rows (4x9)
        for r in range(49):
            U[:, r] = U4[:, r]
            # Map V4 rows (16) to V rows (36)
            for v4_row in range(16):
                # v4_row = j*4 + k where j is block row (0-3), k is within block (0-3)
                j = v4_row // 4
                k = v4_row % 4
                v_row = j * 9 + k  # Map to 4x9 indexing
                V[v_row, r] = V4[v4_row, r]
            # W: same 16 rows for 4x4 output
            W[:16, r] = W4[:, r]
        
        # Block 2: columns 4-7 of B
        col_offset = 49
        for r in range(49):
            U[:, col_offset + r] = U4[:, r]
            for v4_row in range(16):
                j = v4_row // 4
                k = v4_row % 4
                v_row = j * 9 + (k + 4)  # Columns 4-7
                V[v_row, col_offset + r] = V4[v4_row, r]
            W[:16, col_offset + r] = W4[:, r]
        
        # Block 3: column 8 of B (schoolbook for 4x4 * 4x1)
        col_offset = 98
        for i in range(4):
            for j in range(4):
                a = i * 4 + j
                b = j * 9 + 8
                c = 8 * 4 + i  # k=8, n=4, i
                U[a, col_offset] = 1
                V[b, col_offset] = 1
                W[c, col_offset] = 1
                col_offset += 1
        
        return U, V, W

    # ------------------------------------------------------------------
    # Step 4: Verify the decomposition
    # ------------------------------------------------------------------
    def check_exact(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the tensor."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((U_rows, V_rows, W_rows))
        for (a, b, c) in triples:
            T_target[a, b, c] = 1.0
        return np.max(np.abs(T_recon - T_target)) < tol

    # Build the full decomposition
    U_full, V_full, W_full = build_full_decomposition()
    
    # Verify
    if not check_exact(U_full, V_full, W_full, tol=1e-8):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((U_rows, rank_school))
        V_school = np.zeros((V_rows, rank_school))
        W_school = np.zeros((W_rows, rank_school))
        for col, (a, b, c) in enumerate(triples):
            U_school[a, col] = 1
            V_school[b, col] = 1
            W_school[c, col] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_U, best_V, best_W = U_full, V_full, W_full
    best_rank = 114

    # ------------------------------------------------------------------
    # Step 5: Try to reduce rank further via ALS (if time permits)
    # ------------------------------------------------------------------
    def als_decompose(target_rank, seed, max_iters=25, tol=1e-7,
                      init_U=None, init_V=None, init_W=None):
        """ALS decomposition of the matmul tensor."""
        rng = np.random.RandomState(seed)
        
        if init_U is not None:
            U = init_U.astype(float).copy()
        else:
            U = rng.randn(U_rows, target_rank) * 0.01
        if init_V is not None:
            V = init_V.astype(float).copy()
        else:
            V = rng.randn(V_rows, target_rank) * 0.01
        if init_W is not None:
            W = init_W.astype(float).copy()
        else:
            W = rng.randn(W_rows, target_rank) * 0.01

        # Precompute flat targets
        T_U_flat = np.zeros((U_rows, V_rows * W_rows))
        T_V_flat = np.zeros((V_rows, U_rows * W_rows))
        T_W_flat = np.zeros((W_rows, U_rows * V_rows))
        for (a, b, c) in triples:
            T_U_flat[a, b * W_rows + c] = 1.0
            T_V_flat[b, a * W_rows + c] = 1.0
            T_W_flat[c, a * V_rows + b] = 1.0

        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Update U
            VW = np.einsum('br,cr->bcr', V, W)
            B = VW.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows)
            G = B @ B.T + 1e-10 * np.eye(target_rank)
            try:
                U = np.linalg.solve(G, B @ T_U_flat.T).T
            except:
                U = (np.linalg.pinv(G) @ (B @ T_U_flat.T)).T

            # Update V
            UW = np.einsum('ar,cr->acr', U, W)
            C = UW.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)
            Gv = C @ C.T + 1e-10 * np.eye(target_rank)
            try:
                V = np.linalg.solve(Gv, C @ T_V_flat.T).T
            except:
                V = (np.linalg.pinv(Gv) @ (C @ T_V_flat.T)).T

            # Update W
            UV = np.einsum('ar,br->abr', U, V)
            D = UV.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)
            Gw = D @ D.T + 1e-10 * np.eye(target_rank)
            try:
                W = np.linalg.solve(Gw, D @ T_W_flat.T).T
            except:
                W = (np.linalg.pinv(Gw) @ (D @ T_W_flat.T)).T

            # Check error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            T_target = np.zeros((U_rows, V_rows, W_rows))
            for (a, b, c) in triples:
                T_target[a, b, c] = 1.0
            err = np.max(np.abs(T_recon - T_target))
            if err < tol:
                break
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err

        return U, V, W, err

    def round_and_check(U, V, W, scales=None):
        """Try rounding to various scales and check exactness."""
        if scales is None:
            scales = [1, 2, 4, 8, 16, 32]
        for scale in scales:
            inv = 1.0 / scale
            U_r = np.round(U * scale) * inv
            V_r = np.round(V * scale) * inv
            W_r = np.round(W * scale) * inv
            if check_exact(U_r, V_r, W_r, tol=1e-8):
                return U_r, V_r, W_r
        return None

    # Try to reduce from 114
    if time.time() < deadline - 25:
        # Precompute for ALS
        T_U_flat = np.zeros((U_rows, V_rows * W_rows))
        T_V_flat = np.zeros((V_rows, U_rows * W_rows))
        T_W_flat = np.zeros((W_rows, U_rows * V_rows))
        for (a, b, c) in triples:
            T_U_flat[a, b * W_rows + c] = 1.0
            T_V_flat[b, a * W_rows + c] = 1.0
            T_W_flat[c, a * V_rows + b] = 1.0

        candidate_ranks = [112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80]
        
        for target_rank in candidate_ranks:
            if time.time() > deadline - 15:
                break

            found = False
            for seed in range(4):
                if time.time() > deadline - 10:
                    break
                if found:
                    break

                U, V, W, err = als_decompose(target_rank, seed, max_iters=20, tol=1e-7)
                
                if err < 1e-6:
                    rounded = round_and_check(U, V, W)
                    if rounded is not None:
                        best_U, best_V, best_W = rounded
                        best_rank = target_rank
                        found = True
                        break

        # Also try structured reduction: use Strassen for 3 blocks of 3 columns
        # 4x4*4x3 + 4x4*4x3 + 4x4*4x3
        # 4x4*4x3 can potentially be done with less than schoolbook
        # But this is speculative; skip for now

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
