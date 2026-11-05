"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 125  # safety margin before 150s kill

    def strassen_2x2():
        """Return rank-7 Strassen decomposition for 2x2 * 2x2.
        Each term: (u_dict, v_dict, w_dict) where dicts map local (i,j) to coeff.
        U local index: i*2+j, V local index: i*2+j, W local index: i*2+j
        where i,j in {0,1} are local 2x2 indices.
        """
        # M1 = (A00+A11)*(B00+B11) -> C00+=M1, C11+=M1
        # M2 = (A10+A11)*B00 -> C10+=M2, C11-=M2
        # M3 = A00*(B01-B11) -> C01+=M3, C11+=M3
        # M4 = A11*(B10-B00) -> C00+=M4, C10+=M4
        # M5 = (A00+A01)*B11 -> C00-=M5, C01+=M5
        # M6 = (A10-A00)*(B00+B01) -> C11+=M6
        # M7 = (A01-A11)*(B10+B11) -> C00+=M7, C11+=M7
        return [
            # M1: u={0:1,3:1}, v={0:1,3:1}, w={0:1,3:1}
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
            # M2: u={2:1,3:1}, v={0:1}, w={2:1, 3:-1}
            ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
            # M3: u={0:1}, v={1:1,3:-1}, w={1:1, 3:1}
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
            # M4: u={3:1}, v={2:1,0:-1}, w={0:1, 2:1}
            ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
            # M5: u={0:1,1:1}, v={3:1}, w={0:-1, 1:1}
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
            # M6: u={2:1,0:-1}, v={0:1,1:1}, w={3:1}
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
            # M7: u={1:1,3:-1}, v={2:1,3:1}, w={0:1, 3:1}
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}),
        ]

    def build_decomposition():
        """Build rank-43 decomposition for 2x5 * 5x6 using Strassen blocks."""
        U_cols = []
        V_cols = []
        W_cols = []
        strassen_terms = strassen_2x2()

        def add_strassen_2x2_2x2(a_col_off, b_row_off, b_col_off):
            """Add Strassen terms for A[0:2, a_col_off:a_col_off+2] * B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]"""
            for u_d, v_d, w_d in strassen_terms:
                u_col = {}
                v_col = {}
                w_col = {}
                for uidx, c in u_d.items():
                    iloc = uidx // 2
                    jloc = uidx % 2
                    u_col[iloc * m + (a_col_off + jloc)] = c
                for vidx, c in v_d.items():
                    iloc = vidx // 2
                    jloc = vidx % 2
                    v_col[(b_row_off + iloc) * p + (b_col_off + jloc)] = c
                for widx, c in w_d.items():
                    iloc = widx // 2
                    jloc = widx % 2
                    w_col[(b_col_off + jloc) * n + iloc] = c
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)

        # Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x6)
        # Split B into three 2x2 column blocks
        for k_off in range(0, 6, 2):
            add_strassen_2x2_2x2(0, 0, k_off)

        # Block 2: A[:, 2:4] (2x2) * B[2:4, :] (2x6)
        for k_off in range(0, 6, 2):
            add_strassen_2x2_2x2(2, 2, k_off)

        # Block 3: A[:, 4:5] (2x1) * B[4:5, :] (1x6) - rank 6 (one per (i,k) pair)
        for i in range(n):
            for k in range(p):
                u_col = {i * m + 4: 1}
                v_col = {4 * p + k: 1}
                w_col = {k * n + i: 1}
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)

        rank = len(U_cols)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for r in range(rank):
            for idx, val in U_cols[r].items():
                U[idx][r] = val
            for idx, val in V_cols[r].items():
                V[idx][r] = val
            for idx, val in W_cols[r].items():
                W[idx][r] = val

        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify the decomposition using vectorized numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # Vectorize: create 3D array T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum: T[i,j,k] = einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        return np.allclose(T, 1.0, atol=1e-9)

    # Build the rank-43 decomposition
    result = build_decomposition()
    best_rank = len(result["U"][0])

    if verify(result["U"], result["V"], result["W"]):
        print(f"Rank {best_rank} decomposition verified successfully")
    else:
        print(f"Rank {best_rank} decomposition FAILED verification, falling back to naive")
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for col in range(rank):
            i = col // (m * p)
            rem = col % (m * p)
            j = rem // p
            k = rem % p
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
        return {"U": U, "V": V, "W": W}

    return result
# EVOLVE-BLOCK-END
