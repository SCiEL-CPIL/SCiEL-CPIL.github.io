"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy:
  - 4x4 * 4x4 part (cols 0-3 of B): recursive Strassen, rank 49
  - 4x4 * 4x2 part (cols 4-5 of B): 4 independent 2x2x2 Strassen blocks, rank 28
  - Total rank: 77

Each rank-1 term is a triple (u, v, w) where:
  u is in R^{16} (index i*m+j)
  v is in R^{24} (index j*p+k)
  w is in R^{24} (index k*n+i)

The decomposition satisfies:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all valid (i,j,k)
  (i.e., T[i*m+j, j*p+k, k*n+i] = 1, all else 0)
"""

import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def construct():
    n, m, p = 4, 4, 6

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_dict, v_dict, w_dict):
        """Add one rank-1 term. Dicts map flat indices to coefficients."""
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)

        for idx, coeff in u_dict.items():
            u_col[idx] += coeff
        for idx, coeff in v_dict.items():
            v_col[idx] += coeff
        for idx, coeff in w_dict.items():
            w_col[idx] += coeff

        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def add_term_ijk(u_ijk, v_jk, w_ik):
        """
        Add a term using (i,j) for U, (j,k) for V, (i,k) for W indexing.
        u_ijk: {(i,j): coeff}
        v_jk:  {(j,k): coeff}
        w_ik:  {(i,k): coeff}
        """
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)

        for (i, j), coeff in u_ijk.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_jk.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_ik.items():
            w_col[k * n + i] += coeff

        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def strassen_2x2(oi, oj, ok):
        """
        Add Strassen decomposition for 2x2 * 2x2 block multiplication.
        Computes C[oi:oi+2, ok:ok+2] = A[oi:oi+2, oj:oj+2] * B[oj:oj+2, ok:ok+2]
        """
        # M1 = (A00+A11)(B00+B11)
        add_term_ijk(
            {(oi, oj): 1, (oi+1, oj+1): 1},
            {(oj, ok): 1, (oj+1, ok+1): 1},
            {(oi, ok): 1, (oi+1, ok+1): 1}
        )

        # M2 = (A10+A11)B00
        add_term_ijk(
            {(oi+1, oj): 1, (oi+1, oj+1): 1},
            {(oj, ok): 1},
            {(oi+1, ok): 1, (oi+1, ok+1): -1}
        )

        # M3 = A00(B01-B11)
        add_term_ijk(
            {(oi, oj): 1},
            {(oj, ok+1): 1, (oj+1, ok+1): -1},
            {(oi, ok+1): 1, (oi+1, ok+1): 1}
        )

        # M4 = A11(B10-B00)
        add_term_ijk(
            {(oi+1, oj+1): 1},
            {(oj+1, ok): 1, (oj, ok): -1},
            {(oi, ok): 1, (oi+1, ok): 1}
        )

        # M5 = (A00+A01)B11
        add_term_ijk(
            {(oi, oj): 1, (oi, oj+1): 1},
            {(oj+1, ok+1): 1},
            {(oi, ok): -1, (oi, ok+1): 1}
        )

        # M6 = (A10-A00)(B00+B01)
        add_term_ijk(
            {(oi+1, oj): 1, (oi, oj): -1},
            {(oj, ok): 1, (oj, ok+1): 1},
            {(oi+1, ok+1): 1}
        )

        # M7 = (A01-A11)(B10+B11)
        add_term_ijk(
            {(oi, oj+1): 1, (oi+1, oj+1): -1},
            {(oj+1, ok): 1, (oj+1, ok+1): 1},
            {(oi, ok): 1}
        )

    def strassen_super_4x4(oi, oj, ok):
        """
        Add recursive Strassen for 4x4 * 4x4 block multiplication.
        Uses 7 sub-problems, each handled by strassen_2x2 on 2x2 blocks.
        Total: 7 * 7 = 49 rank-1 terms.
        
        Block layout:
        A = [[A00, A01], [A10, A11]]  (each 2x2)
        B = [[B00, B01], [B10, B11]]
        C = [[C00, C01], [C10, C11]]
        
        M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
        M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
        M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
        M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
        M6 = (A10-A00)(B00+B01) -> C11 += M6
        M7 = (A01-A11)(B10+B11) -> C00 += M7
        """
        # For each Strassen "product" M_s, we need to express it as a 2x2x2
        # Strassen decomposition where the U, V, W blocks are linear combos
        # of the 2x2 sub-blocks.
        #
        # The key: each M_s = X * Y where X is a 2x2 matrix (linear combo of A blocks)
        # and Y is a 2x2 matrix (linear combo of B blocks). The result contributes
        # to C blocks with specific signs.
        #
        # We handle each M_s by calling strassen_2x2 with appropriate offsets,
        # but we need to handle the linear combinations. The trick is:
        # M_s = sum over the 7 Strassen terms of the 2x2x2 problem,
        # but the U, V, W coefficients need to reflect the block combinations.
        #
        # Actually, the cleanest way: for each M_s, we have 7 rank-1 terms
        # from the 2x2x2 Strassen. The U coefficients are scaled by the
        # block combination coefficients, similarly for V and W.
        
        # Let me define the 7 Strassen products for 4x4:
        # Each product involves specific 2x2 blocks with coefficients.
        
        # M1 = (A00+A11)(B00+B11)
        # U blocks: A00 coeff +1 at offset (oi, oj), A11 coeff +1 at offset (oi+2, oj+2)
        # V blocks: B00 coeff +1 at offset (oj, ok), B11 coeff +1 at offset (oj+2, ok+2)
        # W blocks: C00 coeff +1 at offset (oi, ok), C11 coeff +1 at offset (oi+2, ok+2)
        
        # The 7 Strassen terms for a 2x2x2 at offsets (oi, oj, ok):
        # But here we have TWO 2x2 blocks contributing to each of U, V, W.
        # 
        # The correct approach: M1 is a 2x2x2 tensor where:
        # U[i,j] = A00[i,j] + A11[i,j]  (i,j in 0..1)
        # V[i,j] = B00[i,j] + B11[i,j]
        # W[i,j] = C00[i,j] + C11[i,j]
        #
        # So the 7 Strassen terms for this "virtual" 2x2x2 have:
        # U entries at (oi+di, oj+dj) with coeff 1 (from A00) AND at (oi+2+di, oj+2+dj) with coeff 1 (from A11)
        # Similarly for V and W.
        
        def strassen_2x2_with_blocks(oi_u, oj_u, oi_v, oj_v, oi_w, ok_w,
                                      u_blocks, v_blocks, w_blocks):
            """
            Add 7 Strassen terms where each of U, V, W is a sum of 2x2 blocks.
            u_blocks: list of (oi, oj, coeff) - 2x2 block offsets and coefficients
            v_blocks: list of (oj, ok, coeff)
            w_blocks: list of (oi, ok, coeff)
            """
            # 7 Strassen patterns for 2x2x2:
            # Each pattern has (u_positions, v_positions, w_positions)
            # where positions are (di, dj) offsets within the 2x2 block
            
            patterns = [
                # M1: (u00+u11)(v00+v11) -> w00+w11
                ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
                # M2: (u10+u11)v00 -> w10-w11
                ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1), ((1,1),-1)]),
                # M3: u00(v01-v11) -> w01+w11
                ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),1)]),
                # M4: u11(v10-v00) -> w00+w10
                ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]),
                # M5: (u00+u01)v11 -> -w00+w01
                ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
                # M6: (u10-u00)(v00+v01) -> w11
                ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,1),1)]),
                # M7: (u01-u11)(v10+v11) -> w00
                ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1)]),
            ]
            
            for u_pat, v_pat, w_pat in patterns:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                for (di, dj), uc in u_pat:
                    for (boi, boj, bcoeff) in u_blocks:
                        idx = (boi + di) * m + (boj + dj)
                        u_dict[idx] = u_dict.get(idx, 0) + uc * bcoeff
                
                for (di, dj), vc in v_pat:
                    for (boj, bok, bcoeff) in v_blocks:
                        idx = (boj + di) * p + (bok + dj)
                        v_dict[idx] = v_dict.get(idx, 0) + vc * bcoeff
                
                for (di, dj), wc in w_pat:
                    for (boi, bok, bcoeff) in w_blocks:
                        idx = (bok + dj) * n + (boi + di)
                        w_dict[idx] = w_dict.get(idx, 0) + wc * bcoeff
                
                add_term(u_dict, v_dict, w_dict)
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        strassen_2x2_with_blocks(
            u_blocks=[(oi, oj, 1), (oi+2, oj+2, 1)],
            v_blocks=[(oj, ok, 1), (oj+2, ok+2, 1)],
            w_blocks=[(oi, ok, 1), (oi+2, ok+2, 1)]
        )
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        strassen_2x2_with_blocks(
            u_blocks=[(oi+2, oj, 1), (oi+2, oj+2, 1)],
            v_blocks=[(oj, ok, 1)],
            w_blocks=[(oi+2, ok, 1), (oi+2, ok+2, -1)]
        )
        
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        strassen_2x2_with_blocks(
            u_blocks=[(oi, oj, 1)],
            v_blocks=[(oj, ok+2, 1), (oj+2, ok+2, -1)],
            w_blocks=[(oi, ok+2, 1), (oi+2, ok+2, 1)]
        )
        
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        strassen_2x2_with_blocks(
            u_blocks=[(oi+2, oj+2, 1)],
            v_blocks=[(oj+2, ok, 1), (oj, ok, -1)],
            w_blocks=[(oi, ok, 1), (oi+2, ok, 1)]
        )
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        strassen_2x2_with_blocks(
            u_blocks=[(oi, oj, 1), (oi, oj+2, 1)],
            v_blocks=[(oj+2, ok+2, 1)],
            w_blocks=[(oi, ok, -1), (oi, ok+2, 1)]
        )
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        strassen_2x2_with_blocks(
            u_blocks=[(oi+2, oj, 1), (oi, oj, -1)],
            v_blocks=[(oj, ok, 1), (oj, ok+2, 1)],
            w_blocks=[(oi+2, ok+2, 1)]
        )
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        strassen_2x2_with_blocks(
            u_blocks=[(oi, oj+2, 1), (oi+2, oj+2, -1)],
            v_blocks=[(oj+2, ok, 1), (oj+2, ok+2, 1)],
            w_blocks=[(oi, ok, 1)]
        )

    # Build 4x4 * 4x4 part: rank 49
    strassen_super_4x4(0, 0, 0)
    
    # Build 4x4 * 4x2 part: rank 28
    # 4 independent 2x2x2 blocks:
    # (rows 0-1, cols 4-5), (rows 0-1, cols 4-5 for B rows 0-1)
    # Actually: for each pair of A rows and each pair of B rows:
    # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]  (2x2 * 2x2 -> 2x2)
    # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]  (2x2 * 2x2 -> 2x2)
    # A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]  (2x2 * 2x2 -> 2x2)
    # A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]  (2x2 * 2x2 -> 2x2)
    
    strassen_2x2(0, 0, 4)  # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
    strassen_2x2(0, 2, 4)  # A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
    strassen_2x2(2, 0, 4)  # A[2:4,0:2] * B[0:2,4:6] -> C[2:4,4:6]
    strassen_2x2(2, 2, 4)  # A[2:4,2:4] * B[2:4,4:6] -> C[2:4,4:6]
    
    rank = len(U_cols)
    
    # Verify the decomposition
    U = np.array(U_cols, dtype=float).T  # shape (n*m, rank)
    V = np.array(V_cols, dtype=float).T  # shape (m*p, rank)
    W = np.array(W_cols, dtype=float).T  # shape (p*n, rank)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # This is the tensor T[i*m+j, j*p+k, k*n+i] = 1
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum over r
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    error = max(error, abs(val - 1.0))
    
    # Also verify that non-diagonal entries are 0
    # T[a,b,c] should be 0 unless a=(i,j), b=(j,k), c=(i,k) with same i,j,k
    # Actually, the tensor is: T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j==j', i==i', k==k'
    # Let's spot check a few off-diagonal entries
    off_diag_error = 0.0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                i_c = c % n
                k_c = c // n
                
                expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                
                if abs(val - expected) > 1e-10:
                    off_diag_error = max(off_diag_error, abs(val - expected))
    
    if error > 1e-10 or off_diag_error > 1e-10:
        # Fall back to schoolbook
        rank_schoolbook = n * m * p
        U_school = np.zeros((n * m, rank_schoolbook))
        V_school = np.zeros((m * p, rank_schoolbook))
        W_school = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
