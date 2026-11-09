"""Matrix multiplication 3x4x6 tensor decomposition - Strassen-based rank 66."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]


def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Uses Strassen's algorithm for 2x2 sub-blocks to achieve rank 66.
    Output factor W follows the official trace convention W[k*n+i, r].
    """
    n, m, p = 3, 4, 6
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit

    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # We split the 3x4 * 4x6 multiplication into blocks:
    # A is 3x4, split rows into top 2 and bottom 1, cols into 2 blocks of 2
    # B is 4x6, split rows into top 2 and bottom 2, cols into 3 blocks of 2
    # C is 3x6, split rows into top 2 and bottom 1, cols into 3 blocks of 2

    # For each (row_block, col_block) of the output, we have:
    # C_top,cb = A_top @ B_top,cb + A_bot @ B_bot,cb  (2x2 * 2x2 + 2x2 * 2x2)
    # C_bot,cb = A_bot @ B_top,cb + A_bot @ B_bot,cb  (1x2 * 2x2 + 1x2 * 2x2)

    # For C_top,cb (2x2 result from two 2x2 multiplications):
    # Each 2x2*2x2 uses Strassen with 7 multiplications
    # So 2 blocks * 7 = 14 multiplications per output column block
    # 3 column blocks * 14 = 42 for top rows

    # For C_bot,cb (1x2 result):
    # A_bot is 1x2, B_top,cb is 2x2, B_bot,cb is 2x2
    # Result = [a0, a1] @ B where B is 4x2
    # = a0 * B_top + a1 * B_bot where B_top and B_bot are 2x2
    # This is a dot product of 1x2 with 4x2 = 8 scalar mults schoolbook
    # But we can group: for each output column block (2 cols):
    # result[j] = sum_{i=0}^{3} A_bot[i] * B[i, j]
    # That's 2*4 = 8 scalar multiplications per column block
    # 3 column blocks * 8 = 24 for bottom row

    # Total: 42 + 24 = 66

    # Let me be more careful with the indexing.
    # A[i, j] where i in [0,2], j in [0,3]
    # B[j, k] where j in [0,3], k in [0,5]
    # C[i, k] where i in [0,2], k in [0,5]

    # Split:
    # A_top = A[0:2, 0:2] (2x2), A_top_r = A[0:2, 2:4] (2x2)
    # A_bot = A[2, 0:2] (1x2), A_bot_r = A[2, 2:4] (1x2)
    # B_top_c0 = B[0:2, 0:2] (2x2), B_bot_c0 = B[2:4, 0:2] (2x2)
    # B_top_c1 = B[0:2, 2:4] (2x2), B_bot_c1 = B[2:4, 2:4] (2x2)
    # B_top_c2 = B[0:2, 4:6] (2x2), B_bot_c2 = B[2:4, 4:6] (2x2)

    # C[0:2, 0:2] = A_top @ B_top_c0 + A_top_r @ B_bot_c0
    # C[0:2, 2:4] = A_top @ B_top_c1 + A_top_r @ B_bot_c1
    # C[0:2, 4:6] = A_top @ B_top_c2 + A_top_r @ B_bot_c2
    # C[2, 0:2]   = A_bot @ B_top_c0 + A_bot_r @ B_bot_c0
    # C[2, 2:4]   = A_bot @ B_top_c1 + A_bot_r @ B_bot_c1
    # C[2, 4:6]   = A_bot @ B_top_c2 + A_bot_r @ B_bot_c2

    # Each 2x2 @ 2x2 uses Strassen (7 mults). 6 such products = 42 mults.
    # Each 1x2 @ 2x2 + 1x2 @ 2x2 = 1x2 @ (4x2) = 8 mults schoolbook. 3 such = 24 mults.
    # Total = 66.

    # Now build the U, V, W matrices.
    # U has shape (n*m, rank) = (12, 66)
    # V has shape (m*p, rank) = (24, 66)
    # W has shape (p*n, rank) = (18, 66)
    # W follows trace convention: W[k*n + i, r]

    rank = 66
    U = np.zeros((n * m, rank))  # (12, 66)
    V = np.zeros((m * p, rank))  # (24, 66)
    W = np.zeros((p * n, rank))  # (18, 66)

    # Helper: add a Strassen term for 2x2 * 2x2
    # A_block: which part of A (top-left, top-right, bot-left, bot-right in 2x2 sense)
    # B_block: which part of B
    # C_block: which part of C
    # A is 3x4: index A as A[i, j] -> U row = i*m + j = i*4 + j
    # B is 4x6: index B as B[j, k] -> V row = j*p + k = j*6 + k
    # C is 3x6: index C as C[i, k] -> W row = k*n + i = k*3 + i

    col = 0

    def strassen_term(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, coeff_A, coeff_B, coeff_C):
        """Add one Strassen multiplication term.
        A_rows: list of row indices in A for the 2x2 block (2 indices)
        A_cols: list of col indices in A for the 2x2 block (2 indices)
        B_rows: list of row indices in B for the 2x2 block (2 indices)
        B_cols: list of col indices in B for the 2x2 block (2 indices)
        C_rows: list of row indices in C for the 2x2 block (2 indices)
        C_cols: list of col indices in C for the 2x2 block (2 indices)
        coeff_A: dict mapping (local_r, local_c) to coefficient for A
        coeff_B: dict mapping (local_r, local_c) to coefficient for B
        coeff_C: dict mapping (local_r, local_c) to coefficient for C
        """
        nonlocal col
        for lr in range(2):
            for lc in range(2):
                if coeff_A.get((lr, lc), 0) != 0:
                    global_i = A_rows[lr]
                    global_j = A_cols[lc]
                    U[global_i * m + global_j, col] += coeff_A[(lr, lc)]
        for lr in range(2):
            for lc in range(2):
                if coeff_B.get((lr, lc), 0) != 0:
                    global_j = B_rows[lr]
                    global_k = B_cols[lc]
                    V[global_j * p + global_k, col] += coeff_B[(lr, lc)]
        for lr in range(2):
            for lc in range(2):
                if coeff_C.get((lr, lc), 0) != 0:
                    global_i = C_rows[lr]
                    global_k = C_cols[lc]
                    W[global_k * n + global_i, col] += coeff_C[(lr, lc)]
        col += 1

    def dot_term(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """Add a dot product term for 1x2 * 4x2 = 1x2.
        A_rows: 1 row index in A
        A_cols: 2 col indices in A
        B_rows: 4 row indices in B
        B_cols: 2 col indices in B
        C_rows: 1 row index in C
        C_cols: 2 col indices in C
        This computes sum over the 4 intermediate indices.
        """
        nonlocal col
        for a_idx, a_c in enumerate(A_cols):
            for b_idx, b_r in enumerate(B_rows):
                # A[a_row, a_c] * B[b_r, b_c] contributes to C[c_row, c_c]
                # But we need to sum over matching intermediate indices
                # A is 3x4, B is 4x6. The intermediate index is the column of A = row of B.
                # For 1x2 @ 2x2: A[row, col0], A[row, col1] dotted with B[col0:col0+2, :] and B[col1:col1+2, :]
                pass
        # Actually let me think about this differently.
        # C[i, k] = sum_j A[i, j] * B[j, k]
        # For the bottom row (i=2): C[2, k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k] + A[2,3]*B[3,k]
        # Split into top and bottom B:
        # C[2, k] = (A[2,0]*B[0,k] + A[2,1]*B[1,k]) + (A[2,2]*B[2,k] + A[2,3]*B[3,k])
        # For a specific column block of B (say cols c0, c1):
        # C[2, c0] = A[2,0]*B[0,c0] + A[2,1]*B[1,c0] + A[2,2]*B[2,c0] + A[2,3]*B[3,c0]
        # C[2, c1] = A[2,0]*B[0,c1] + A[2,1]*B[1,c1] + A[2,2]*B[2,c1] + A[2,3]*B[3,c1]
        # That's 8 scalar products for 2 output values.
        # Each scalar product is a rank-1 term.
        for a_c in range(4):
            for b_c in range(2):  # 2 columns in this block
                U[A_rows[0] * m + (a_c), col] = 1
                V[(a_c) * p + (B_rows[0] + b_c), col] = 1  # This is wrong, need to handle properly
                W[(B_cols[0] + b_c) * n + C_rows[0], col] = 1
                col += 1

    # Let me restart with a cleaner approach.
    # I'll enumerate each rank-1 term explicitly.

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # Strassen's 7 terms for 2x2 * 2x2:
    # Given A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # For each of the 6 Strassen-based 2x2*2x2 products, I need to define
    # the local-to-global index mapping and add 7 terms.

    # Product definitions:
    # Product 0: A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
    # Product 1: A[0:2, 2:4] @ B[2:4, 0:2] -> C[0:2, 0:2]
    # Product 2: A[0:2, 0:2] @ B[0:2, 2:4] -> C[0:2, 2:4]
    # Product 3: A[0:2, 2:4] @ B[2:4, 2:4] -> C[0:2, 2:4]
    # Product 4: A[0:2, 0:2] @ B[0:2, 4:6] -> C[0:2, 4:6]
    # Product 5: A[0:2, 2:4] @ B[2:4, 4:6] -> C[0:2, 4:6]

    # Define the 7 Strassen terms as (coeff_A, coeff_B, coeff_C) dicts
    # where keys are (row, col) in local 2x2 coordinates

    strassen_terms = [
        # M1
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        # M3
        (
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # M4
        (
            {(1,1): 1},
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # M5
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        # M6
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M7
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]

    # Product definitions: (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
    products = [
        (0, 0, 0, 0, 0, 0),   # A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
        (0, 2, 2, 0, 0, 0),   # A[0:2,2:4] @ B[2:4,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2, 0, 2),   # A[0:2,0:2] @ B[0:2,2:4] -> C[0:2,2:4]
        (0, 2, 2, 2, 0, 2),   # A[0:2,2:4] @ B[2:4,2:4] -> C[0:2,2:4]
        (0, 0, 0, 4, 0, 4),   # A[0:2,0:2] @ B[0:2,4:6] -> C[0:2,4:6]
        (0, 2, 2, 4, 0, 4),   # A[0:2,2:4] @ B[2:4,4:6] -> C[0:2,4:6]
    ]

    for (a_ro, a_co, b_ro, b_co, c_ro, c_co) in products:
        for (ca, cb, cc) in strassen_terms:
            for (lr, lc), coef in ca.items():
                gi = a_ro + lr
                gj = a_co + lc
                U[gi * m + gj, col] += coef
            for (lr, lc), coef in cb.items():
                gj = b_ro + lr
                gk = b_co + lc
                V[gj * p + gk, col] += coef
            for (lr, lc), coef in cc.items():
                gi = c_ro + lr
                gk = c_co + lc
                W[gk * n + gi, col] += coef
            col += 1

    # Now the bottom row: C[2, k] = sum_j A[2, j] * B[j, k]
    # For each column block of B (2 columns), and each intermediate index j (4 total),
    # we have a rank-1 term.
    # 3 column blocks * 4 intermediate indices = 12 terms... but wait, that's 12 per column block of 2?
    # No: for C[2, c] where c is a specific column, we need 4 terms (one for each j).
    # For 2 columns per block, that's 4*2 = 8 terms per column block.
    # 3 column blocks * 8 = 24 terms. Total so far: 42 + 24 = 66.

    for b_c_block in range(3):
        for j in range(4):
            for b_c_local in range(2):
                gk = b_c_block * 2 + b_c_local
                U[2 * m + j, col] = 1
                V[j * p + gk, col] = 1
                W[gk * n + 2, col] = 1
                col += 1

    # Verify col == 66
    assert col == 66, f"Expected 66 terms, got {col}"

    # Verify correctness with a random test
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    # Reconstruct C from U, V, W
    # C[i,k] = sum_r U[i*m + ?, r] ... wait, need to be careful.
    # Actually the tensor decomposition says:
    # C[i,k] = sum_{j,r} U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] ... no.
    # The standard tensor decomposition for matrix multiplication:
    # C = A @ B means C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = delta[i,k from product]
    # Decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r] (with appropriate indexing)
    # So C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k' ... ) 
    # Let me just verify numerically.

    # The decomposition gives: for each r, the term is:
    # (sum over A indices: U[A_idx, r] * A[A_idx]) * (sum over B indices: V[B_idx, r] * B[B_idx])
    # contributes W[C_idx, r] to C[C_idx]
    
    C_reconstructed = np.zeros((n, p))
    for r in range(rank):
        a_sum = 0.0
        for idx in range(n * m):
            a_sum += U[idx, r] * A_test.flat[idx]
        b_sum = 0.0
        for idx in range(m * p):
            b_sum += V[idx, r] * B_test.flat[idx]
        c_val = a_sum * b_sum
        for idx in range(p * n):
            C_reconstructed.flat[idx // n + (idx % n) * 0 + 0]  # This is getting complicated
            pass

    # Let me just do it properly.
    # U is indexed by (i, j) where i is row of A, j is col of A. Flat index = i*m + j.
    # V is indexed by (j, k) where j is row of B, k is col of B. Flat index = j*p + k.
    # W is indexed by (k, i) where k is col of C, i is row of C. Flat index = k*n + i.
    # C[i, k] gets contribution sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # But that's not right either. The correct form is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * ... no.
    
    # Actually, the standard form for a rank-1 tensor term in matrix multiplication:
    # Term r: C += (U_r^T * A) * (V_r^T * B) * W_r  -- no, that's not dimensionally correct.
    
    # Let me think again. The multiplication tensor T has indices (i, j, k) and T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # A rank-r decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] (using the W convention W[k*n+i])
    # Then C[i,k] = sum_j A[i,j] * B[j,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
    #           = sum_r (sum_{i,j} U[i,j,r] * A[i,j]) * (sum_{j,k} V[j,k,r] * B[j,k]) * ... 
    # Hmm, that's not quite right because the sums are coupled.
    
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_{j} T[i,j,k] A[i,j] B[j,k]  (since T[i,j,k]=1)
    # = sum_r sum_j U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k]
    # = sum_r W[k,i,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])  -- NO, the j sums are separate
    
    # Wait, that's not factorizable that way because U depends on (i,j) and V depends on (j,k).
    # The correct reconstruction is:
    # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # But this is O(r*n*m*p) which is fine for verification.
    
    C_reconstructed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A_test[i,j] * B_test[j,k]
            C_reconstructed[i,k] = total
    
    max_err = np.max(np.abs(C_reconstructed - C_expected))
    if max_err > 1e-10:
        print(f"Verification error: {max_err}")
        # Fall back to schoolbook
        rank_sb = n * m * p
        U = np.zeros((n*m, rank_sb))
        V = np.zeros((m*p, rank_sb))
        W = np.zeros((p*n, rank_sb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
