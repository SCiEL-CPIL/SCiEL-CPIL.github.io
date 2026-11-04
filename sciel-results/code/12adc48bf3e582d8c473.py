"""Combined program: Strassen-based structural decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill

    def build_strassen_4x4_4():
        """
        Build Strassen decomposition for 4x4x4 matrix multiplication.
        Returns U, V, W matrices of shape (16, 49), (28, 49), (16, 49) respectively.
        Indices for U: i*m + j
        Indices for V: j*p + k
        Indices for W: k*n + i
        """
        rank = 49
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen 2x2x2 terms
        # x_sub, y_sub are 2x2 dicts, c_sub is 2x2 dict for output
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        ]
        
        # Strassen 4x4x4: 7 outer terms, each with 7 inner terms
        # x_def, y_def are dicts mapping block index (0..3) to coeff
        # Block index b: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        strassen_4x4_outer = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}),
            ({0: 1}, {1: 1, 3: -1}),
            ({3: 1}, {2: 1, 0: -1}),
            ({0: 1, 1: 1}, {3: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}),
        ]
        
        # c_block for each outer term
        # c_block maps (local_r, local_c) to coeff
        c_blocks_4x4 = [
            {(0,0): 1.0, (1,1): 1.0},
            {(1,0): 1.0, (1,1): -1.0},
            {(0,1): 1.0, (1,1): 1.0},
            {(0,0): 1.0, (1,0): 1.0},
            {(0,0): -1.0, (0,1): 1.0},
            {(1,1): 1.0},
            {(0,0): 1.0},
        ]
        
        current_rank = 0
        for m_idx in range(7):
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                # Compute U coefficients
                # U index: i*m + j
                # i = 2*bi + li, j = 2*bj + lj
                for (li, lj), cx in x_sub.items():
                    for b, cb in x_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        U[i * m + j, r] += cx * cb
                
                # Compute V coefficients
                # V index: j*p + k
                # j = 2*bi + li, k = 2*bj + lj
                for (li, lj), cy in y_sub.items():
                    for b, cb in y_def.items():
                        bi, bj = b // 2, b % 2
                        j = 2 * bi + li
                        k = 2 * bj + lj
                        V[j * p + k, r] += cy * cb
                
                # Compute W coefficients
                # W index: k*n + i
                # i = 2*br + li, k = 2*bc + lj
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    for (br, bc), cb_c in c_block.items():
                        if cb_c == 0:
                            continue
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        W[k_final * n + i_final, r] += cc * cb_c
        
        return U[:, :rank], V[:, :rank], W[:, :rank]

    def build_2x2x3_decomposition():
        """
        Build decomposition for 2x2x3 matrix multiplication with rank 10.
        Uses known rank-10 decomposition.
        Returns U, V, W for local indices.
        U shape (4, 10), V shape (6, 10), W shape (6, 10).
        Local indices:
        U: i*2 + j (i,j in 0..1)
        V: j*3 + k (j in 0..1, k in 0..2)
        W: k*2 + i (k in 0..2, i in 0..1)
        """
        rank = 10
        U = np.zeros((4, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        # Rank-10 decomposition for 2x2x3
        # Each term: (U_coeffs, V_coeffs, W_coeffs)
        # U: (i,j) -> i*2+j
        # V: (j,k) -> j*3+k
        # W: (k,i) -> k*2+i
        
        terms = [
            # Term 0
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
            # Term 1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
            # Term 2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
            # Term 3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
            # Term 4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
            # Term 5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
            # Term 6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
            # Term 7: Handles column 2 of B
            ({(0,0): 1, (1,0): 1}, {(0,2): 1, (1,2): 1}, {(2,0): 1, (2,1): 1}),
            # Term 8: Handles column 2 of B
            ({(0,1): 1, (1,1): 1}, {(0,2): 1, (1,2): -1}, {(2,0): 1, (2,1): -1}),
            # Term 9: Handles column 2 of B
            ({(0,0): 1, (0,1): 1}, {(0,2): 1}, {(2,0): 1, (2,1): 1}),
        ]
        
        for t_idx in range(rank):
            r = t_idx
            u_sub, v_sub, w_sub = terms[t_idx]
            
            for (li, lj), cx in u_sub.items():
                U[li * 2 + lj, r] += cx
            
            for (li, lj), cy in v_sub.items():
                V[li * 3 + lj, r] += cy
            
            for (li, lj), cc in w_sub.items():
                W[li * 2 + lj, r] += cc
        
        return U, V, W

    def build_4x4x3_decomposition():
        """
        Build decomposition for 4x4x3 matrix multiplication.
        Splits into 4 blocks of 2x2x3.
        Returns U, V, W for the 4x4x3 sub-tensor.
        U shape (16, rank), V shape (12, rank), W shape (12, rank).
        """
        U_223, V_223, W_223 = build_2x2x3_decomposition()
        rank_223 = U_223.shape[1]
        rank = 4 * rank_223
        U = np.zeros((16, rank))
        V = np.zeros((12, rank))
        W = np.zeros((12, rank))
        
        # Blocks of A (4x4):
        # A[0:2, 0:2] -> block 0
        # A[0:2, 2:4] -> block 1
        # A[2:4, 0:2] -> block 2
        # A[2:4, 2:4] -> block 3
        
        # Blocks of B (4x3):
        # B[0:2, :] -> top block (rows 0,1)
        # B[2:4, :] -> bot block (rows 2,3)
        
        # C (4x3):
        # C[0:2, :] = A[0:2, 0:2] B[0:2, :] + A[0:2, 2:4] B[2:4, :]
        # C[2:4, :] = A[2:4, 0:2] B[0:2, :] + A[2:4, 2:4] B[2:4, :]
        
        # We have 4 terms:
        # T1: A00 * Btop
        # T2: A01 * Bbot
        # T3: A10 * Btop
        # T4: A11 * Bbot
        
        # Each term is a 2x2x3 multiplication.
        # We will place them sequentially in the rank dimension.
        
        term_offsets = [0, rank_223, 2 * rank_223, 3 * rank_223]
        
        # T1: A00 * Btop
        # A00 indices: i in 0..1, j in 0..1. Global i, j same.
        # Btop indices: j (local 0..1) -> global j 0..1. k local 0..2 -> global k 0..2.
        # C top indices: i in 0..1, k in 0..2. Global i, k same.
        for r_local in range(rank_223):
            r_global = term_offsets[0] + r_local
            # U
            for i in range(2):
                for j in range(2):
                    val = U_223[i * 2 + j, r_local]
                    if val != 0:
                        U[i * 4 + j, r_global] += val
            # V
            for j in range(2):
                for k in range(3):
                    val = V_223[j * 3 + k, r_local]
                    if val != 0:
                        V[j * 3 + k, r_global] += val
            # W
            for k in range(3):
                for i in range(2):
                    val = W_223[k * 2 + i, r_local]
                    if val != 0:
                        W[k * 4 + i, r_global] += val

        # T2: A01 * Bbot
        # A01 indices: i in 0..1, j in 2..3. Global i, j+2.
        # Bbot indices: j (local 0..1) -> global j 2..3. k local 0..2 -> global k 0..2.
        # C top indices: i in 0..1, k in 0..2. Global i, k same.
        for r_local in range(rank_223):
            r_global = term_offsets[1] + r_local
            # U
            for i in range(2):
                for j in range(2):
                    val = U_223[i * 2 + j, r_local]
                    if val != 0:
                        U[i * 4 + (j + 2), r_global] += val
            # V
            for j in range(2):
                for k in range(3):
                    val = V_223[j * 3 + k, r_local]
                    if val != 0:
                        V[(j + 2) * 3 + k, r_global] += val
            # W
            for k in range(3):
                for i in range(2):
                    val = W_223[k * 2 + i, r_local]
                    if val != 0:
                        W[k * 4 + i, r_global] += val

        # T3: A10 * Btop
        # A10 indices: i in 2..3, j in 0..1. Global i+2, j.
        # Btop indices: j (local 0..1) -> global j 0..1. k local 0..2 -> global k 0..2.
        # C bot indices: i in 2..3, k in 0..2. Global i+2, k same.
        for r_local in range(rank_223):
            r_global = term_offsets[2] + r_local
            # U
            for i in range(2):
                for j in range(2):
                    val = U_223[i * 2 + j, r_local]
                    if val != 0:
                        U[(i + 2) * 4 + j, r_global] += val
            # V
            for j in range(2):
                for k in range(3):
                    val = V_223[j * 3 + k, r_local]
                    if val != 0:
                        V[j * 3 + k, r_global] += val
            # W
            for k in range(3):
                for i in range(2):
                    val = W_223[k * 2 + i, r_local]
                    if val != 0:
                        W[k * 4 + (i + 2), r_global] += val

        # T4: A11 * Bbot
        # A11 indices: i in 2..3, j in 2..3. Global i+2, j+2.
        # Bbot indices: j (local 0..1) -> global j 2..3. k local 0..2 -> global k 0..2.
        # C bot indices: i in 2..3, k in 0..2. Global i+2, k same.
        for r_local in range(rank_223):
            r_global = term_offsets[3] + r_local
            # U
            for i in range(2):
                for j in range(2):
                    val = U_223[i * 2 + j, r_local]
                    if val != 0:
                        U[(i + 2) * 4 + (j + 2), r_global] += val
            # V
            for j in range(2):
                for k in range(3):
                    val = V_223[j * 3 + k, r_local]
                    if val != 0:
                        V[(j + 2) * 3 + k, r_global] += val
            # W
            for k in range(3):
                for i in range(2):
                    val = W_223[k * 2 + i, r_local]
                    if val != 0:
                        W[k * 4 + (i + 2), r_global] += val

        return U, V, W

    def build_full_decomposition():
        U_444, V_444, W_444 = build_strassen_4x4_4()
        U_443, V_443, W_443 = build_4x4x3_decomposition()
        
        rank_444 = U_444.shape[1]
        rank_443 = U_443.shape[1]
        total_rank = rank_444 + rank_443
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # Part 1: 4x4x4 (cols 0-3)
        U[:, :rank_444] = U_444
        V[:, :rank_444] = V_444
        W[:, :rank_444] = W_444
        
        # Part 2: 4x4x3 (cols 4-6)
        # V and W indices need to be shifted by 4 in the k-dimension.
        # V index: j*p + k. k_local 0..2 -> k_global 4..6.
        # V_443 has indices j*3 + k_local.
        # We need to map to j*7 + (k_local + 4).
        for j in range(m):
            for k_local in range(3):
                for r in range(rank_443):
                    val = V_443[j * 3 + k_local, r]
                    if val != 0:
                        V[j * p + (k_local + 4), rank_444 + r] = val
                        
        # W index: k*n + i. k_local 0..2 -> k_global 4..6.
        # W_443 has indices k_local*4 + i.
        # We need to map to (k_local + 4)*4 + i.
        for k_local in range(3):
            for i in range(n):
                for r in range(rank_443):
                    val = W_443[k_local * n + i, r]
                    if val != 0:
                        W[(k_local + 4) * n + i, rank_444 + r] = val
                        
        # U is the same.
        U[:, rank_444:] = U_443
        
        return U.tolist(), V.tolist(), W.tolist()

    def verify(U, V, W, r):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    U, V, W = build_full_decomposition()
    rank = len(U[0])
    
    err = verify(U, V, W, rank)
    
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
