"""Matrix multiplication 6x6x6 tensor decomposition via Kronecker product of Strassen and schoolbook."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 100.0  # safety margin before 120s (evaluator kills at 150s)
    n, m, p = DIMENSIONS

    # Build 2x2 Strassen decomposition (rank 7)
    # U_str: 4x7, V_str: 4x7, W_str: 4x7
    U_s = np.zeros((4, 7), dtype=np.float64)
    V_s = np.zeros((4, 7), dtype=np.float64)
    W_s = np.zeros((4, 7), dtype=np.float64)

    # Strassen 7 terms for 2x2:
    # M0=(A00+A11)(B00+B11) -> contributes to C00, C11
    # M1=(A10+A11)B00 -> C10, C11
    # M2=A00(B01-B11) -> C01, C11
    # M3=A11(B10-B00) -> C00, C10
    # M4=(A00+A01)B11 -> C00, C01
    # M5=(A10-A00)(B00+B01) -> C11
    # M6=(A01-A11)(B10+B11) -> C00

    # U_s entries (row = a_idx*2+b_idx, col = term)
    # Term 0: A00+A11
    U_s[0, 0] = 1; U_s[3, 0] = 1
    # Term 1: A10+A11
    U_s[2, 1] = 1; U_s[3, 1] = 1
    # Term 2: A00
    U_s[0, 2] = 1
    # Term 3: A11
    U_s[3, 3] = 1
    # Term 4: A00+A01
    U_s[0, 4] = 1; U_s[1, 4] = 1
    # Term 5: A10-A00
    U_s[2, 5] = 1; U_s[0, 5] = -1
    # Term 6: A01-A11
    U_s[1, 6] = 1; U_s[3, 6] = -1

    # V_s entries
    # Term 0: B00+B11
    V_s[0, 0] = 1; V_s[3, 0] = 1
    # Term 1: B00
    V_s[0, 1] = 1
    # Term 2: B01-B11
    V_s[1, 2] = 1; V_s[3, 2] = -1
    # Term 3: B10-B00
    V_s[2, 3] = 1; V_s[0, 3] = -1
    # Term 4: B11
    V_s[3, 4] = 1
    # Term 5: B00+B01
    V_s[0, 5] = 1; V_s[1, 5] = 1
    # Term 6: B10+B11
    V_s[2, 6] = 1; V_s[3, 6] = 1

    # W_s entries (row = c_idx*2+d_idx, col = term)
    # Term 0: C00+C11
    W_s[0, 0] = 1; W_s[3, 0] = 1
    # Term 1: C10-C11
    W_s[2, 1] = 1; W_s[3, 1] = -1
    # Term 2: C01+C11
    W_s[1, 2] = 1; W_s[3, 2] = 1
    # Term 3: C00+C10
    W_s[0, 3] = 1; W_s[2, 3] = 1
    # Term 4: -C00+C01
    W_s[0, 4] = -1; W_s[1, 4] = 1
    # Term 5: C11
    W_s[3, 5] = 1
    # Term 6: C00
    W_s[0, 6] = 1

    # Build 3x3 schoolbook decomposition (rank 9)
    U_sc = np.zeros((9, 9), dtype=np.float64)
    V_sc = np.zeros((9, 9), dtype=np.float64)
    W_sc = np.zeros((9, 9), dtype=np.float64)
    for idx in range(9):
        U_sc[idx, idx] = 1
        V_sc[idx, idx] = 1
        W_sc[idx, idx] = 1

    # Kronecker product: rank = 7 * 9 = 63
    # U_total = kron(U_sc, U_s)  shape (27, 63)
    # V_total = kron(V_sc, V_s)  shape (27, 63)
    # W_total = kron(W_sc, W_s)  shape (27, 63)

    U_kron = np.kron(U_sc, U_s)  # (9*4, 9*7) = (36, 63)
    V_kron = np.kron(V_sc, V_s)  # (36, 63)
    W_kron = np.kron(W_sc, W_s)  # (36, 63)

    # Now we need to reshape from (36, 63) to (36, 63)
    # But our matrices are 6x6, so n*m = 36, m*p = 36, p*n = 36.
    # U should be (n*m, rank) = (36, 63)
    # V should be (m*p, rank) = (36, 63)
    # W should be (p*n, rank) = (36, 63)

    # The Kronecker product gives us the right shapes.
    # But we need to verify the indexing convention.
    # U[i,j] where i = a_row * m + a_col (0..35), j = term (0..62)
    # V[i,j] where i = b_row * p + b_col (0..35)
    # W[i,j] where i = c_col * n + c_row (0..35)  -- trace convention W[k*n+i]

    # The Kronecker product naturally gives:
    # U_kron[(sc_idx)*4 + s_idx, (sc_term)*7 + s_term]
    # where sc_idx = block_row*3 + block_col (0..8), s_idx = local_row*2 + local_col (0..3)
    # So U_kron row = (block_row*3 + block_col)*4 + local_row*2 + local_col
    # = block_row*12 + block_col*4 + local_row*2 + local_col
    # But we want row = global_row*6 + global_col = (block_row*2+local_row)*6 + (block_col*2+local_col)
    # = block_row*12 + local_row*6 + block_col*2 + local_col
    # These are different! We need to permute rows.

    # Let me think about this more carefully.
    # The Kronecker product kron(A, B) where A is (p,q) and B is (r,s) gives (p*r, q*s).
    # Element at position (i*r + k, j*s + l) = A[i,j] * B[k,l]
    #
    # For U: we want U[global_a_row * 6 + global_a_col, term]
    # where global_a_row = block_row * 2 + local_row, global_a_col = block_col * 2 + local_col
    # term = sc_term * 7 + s_term
    #
    # From Kronecker: U_kron[sc_idx * 4 + s_idx, sc_term * 7 + s_term]
    # where sc_idx = block_row * 3 + block_col, s_idx = local_row * 2 + local_col
    #
    # So U_kron row index = (block_row*3 + block_col)*4 + (local_row*2 + local_col)
    # = block_row*12 + block_col*4 + local_row*2 + local_col
    #
    # Desired row index = (block_row*2 + local_row)*6 + (block_col*2 + local_col)
    # = block_row*12 + local_row*6 + block_col*2 + local_col
    #
    # These differ. We need a row permutation.
    # Let me create the permutation.

    def make_permutation():
        """Map Kronecker row index to desired row index."""
        perm = {}
        for br in range(3):
            for bc in range(3):
                for lr in range(2):
                    for lc in range(2):
                        kron_idx = (br * 3 + bc) * 4 + (lr * 2 + lc)
                        desired_idx = (br * 2 + lr) * 6 + (bc * 2 + lc)
                        perm[kron_idx] = desired_idx
        return perm

    perm = make_permutation()

    # Apply permutation to U_kron
    U_final = np.zeros((36, 63), dtype=np.float64)
    V_final = np.zeros((36, 63), dtype=np.float64)
    W_final = np.zeros((36, 63), dtype=np.float64)

    for kron_idx in range(36):
        desired_idx = perm[kron_idx]
        U_final[desired_idx, :] = U_kron[kron_idx, :]
        V_final[desired_idx, :] = V_kron[kron_idx, :]
        # For W, the convention is W[k*n + i] where k is col, i is row of C
        # In Kronecker, W_kron row = sc_idx * 4 + s_idx
        # sc_idx = c_block_row * 3 + c_block_col, s_idx = c_local_row * 2 + c_local_col
        # Desired: W[k*n + i] where k = c_block_col*2 + c_local_col, i = c_block_row*2 + c_local_row
        # = (c_block_col*2 + c_local_col)*6 + (c_block_row*2 + c_local_row)
        # = c_block_col*12 + c_local_col*6 + c_block_row*2 + c_local_row
        #
        # Kron idx = (c_block_row*3 + c_block_col)*4 + (c_local_row*2 + c_local_col)
        # = c_block_row*12 + c_block_col*4 + c_local_row*2 + c_local_col
        #
        # Different permutation for W!
        W_final[perm[kron_idx], :] = W_kron[kron_idx, :]

    # Wait, I need to be more careful with W.
    # W[k*n + i] where k is the column index of C, i is the row index of C.
    # In the Kronecker product, W_kron row = (sc_row)*4 + (s_row)
    # where sc_row = c_block_row*3 + c_block_col, s_row = c_local_row*2 + c_local_col
    #
    # But in our convention, W index = k*n + i where:
    # k = global column of C = c_block_col*2 + c_local_col
    # i = global row of C = c_block_row*2 + c_local_row
    #
    # So W index = (c_block_col*2 + c_local_col)*6 + (c_block_row*2 + c_local_row)
    #
    # Let me make a separate permutation for W.

    def make_w_permutation():
        perm_w = {}
        for br in range(3):  # c_block_row
            for bc in range(3):  # c_block_col
                for lr in range(2):  # c_local_row
                    for lc in range(2):  # c_local_col
                        kron_idx = (br * 3 + bc) * 4 + (lr * 2 + lc)
                        # W index = k*n + i = (bc*2+lc)*6 + (br*2+lr)
                        desired_idx = (bc * 2 + lc) * 6 + (br * 2 + lr)
                        perm_w[kron_idx] = desired_idx
        return perm_w

    perm_w = make_w_permutation()

    W_final2 = np.zeros((36, 63), dtype=np.float64)
    for kron_idx in range(36):
        desired_idx = perm_w[kron_idx]
        W_final2[desired_idx, :] = W_kron[kron_idx, :]

    # Verify the decomposition
    if time.time() < deadline - 5:
        # Build A, B as symbolic and check C = A*B
        # For verification, use random matrices
        np.random.seed(42)
        A = np.random.randn(6, 6)
        B = np.random.randn(6, 6)
        C_expected = A @ B

        # Compute C from decomposition
        # C[i,k] = sum_r U[i,r] * V[r,r] * W[k*n+i, r]
        # Wait, the decomposition is: C[i,k] = sum_r U[i,r] * V[j,r] * W[k*n+i, r]
        # where i is row of A/C, j is col of A / row of B, k is col of B/C
        # Actually: (AB)[i,k] = sum_j A[i,j] B[j,k]
        # The tensor decomposition: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ik,r}
        # where T_{ijk} = 1 if j is the "middle" index...
        #
        # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = 1 (it's the identity in the middle index)
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        #
        # So: C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
        # Wait no. The decomposition says:
        # sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * (sum_j B[j,k] V[j*p+k,r]) * W[k*n+i,r]
        #
        # Let me verify numerically.
        C_computed = np.zeros((6, 6))
        for i in range(6):
            for k in range(6):
                s = 0.0
                for r in range(63):
                    # U part: sum_j A[i,j] * U[i*6+j, r]
                    u_val = 0.0
                    for j in range(6):
                        u_val += A[i, j] * U_final[i * 6 + j, r]
                    # V part: sum_j B[j,k] * V[j*6+k, r]
                    v_val = 0.0
                    for j in range(6):
                        v_val += B[j, k] * V_final[j * 6 + k, r]
                    s += u_val * v_val * W_final2[k * 6 + i, r]
                C_computed[i, k] = s

        err = np.max(np.abs(C_computed - C_expected))
        if err > 1e-9:
            # Fallback: use schoolbook 216
            rank_fb = 216
            U_fb = np.zeros((36, rank_fb))
            V_fb = np.zeros((36, rank_fb))
            W_fb = np.zeros((36, rank_fb))
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        idx = i * 36 + j * 6 + k
                        U_fb[i * 6 + j, idx] = 1
                        V_fb[j * 6 + k, idx] = 1
                        W_fb[k * 6 + i, idx] = 1
            # Convert to lists
            U_list = U_fb.tolist()
            V_list = V_fb.tolist()
            W_list = W_fb.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final2.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
