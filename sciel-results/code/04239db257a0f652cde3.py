"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    def elapsed():
        return time.time() - start_time

    def schoolbook():
        """Rank n*m*p exact decomposition (always valid)."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2(U_rows, V_rows, W_rows, rank_offset):
        """
        Apply Strassen's 7-term decomposition to a 2x2 block multiplication.
        U_rows: list of (i,j) pairs for A's 2x2 block
        V_rows: list of (j,k) pairs for B's 2x2 block  
        W_rows: list of (k,i) pairs for C's 2x2 block
        Returns number of terms used (7)
        """
        # Strassen's 7 products for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map block indices to actual matrix indices
        # For a 2x2 block, we have local indices 0,1 mapping to global
        # The block structure: A is n x m, B is m x p, C is p x n
        # We decompose into 3x3 blocks of sizes [2,2,1]
        
        # This helper will be called for each block pair (a,b) where a is row-block of A, b is col-block of B
        # and the result goes to C block (b, a)
        
        # For now, this is a placeholder - the actual implementation below handles it
        return 7

    def block_strassen_decomposition():
        """
        Block-Strassen decomposition for 5x5 matrix multiplication.
        5 = 2 + 2 + 1, so we have 3x3 = 9 blocks.
        
        A (5x5) is partitioned into blocks A[bi][bj] where bi,bj in {0,1,2}
        B (5x5) is partitioned into blocks B[bj][bk]
        C (5x5) is partitioned into blocks C[bk][bi]
        
        For each block multiplication C[k,i] += A[i,j] * B[j,k],
        we apply Strassen when both blocks are 2x2, otherwise use schoolbook.
        """
        block_sizes = [2, 2, 1]
        num_blocks = 3
        
        # Precompute offsets
        offsets = [0]
        for bs in block_sizes:
            offsets.append(offsets[-1] + bs)
        
        # Collect all terms
        U_terms = []  # Each term is a dict mapping (i,j) -> coefficient
        V_terms = []  # Each term is a dict mapping (j,k) -> coefficient
        W_terms = []  # Each term is a dict mapping (k,i) -> coefficient
        
        for bi in range(num_blocks):  # row block of A, col block of C
            for bj in range(num_blocks):  # col block of A, row block of B
                for bk in range(num_blocks):  # col block of B, row block of C
                    # We need to compute C[bk][bi] += A[bi][bj] * B[bj][bk]
                    
                    size_a = block_sizes[bi] * block_sizes[bj]  # A block is size_a x size_bj
                    size_b = block_sizes[bj] * block_sizes[bk]  # B block is size_bj x size_bk
                    
                    # Get the actual dimensions
                    a_rows = block_sizes[bi]
                    a_cols = block_sizes[bj]
                    b_rows = block_sizes[bj]
                    b_cols = block_sizes[bk]
                    
                    # If both blocks are 2x2 (or involve only 2-sized blocks), use Strassen
                    # For simplicity, we'll use Strassen when a_rows==2 and b_cols==2 and a_cols==2 and b_rows==2
                    # Otherwise, use schoolbook for that block pair
                    
                    if a_rows == 2 and a_cols == 2 and b_rows == 2 and b_cols == 2:
                        # Apply Strassen's 7-term decomposition
                        # Local indices: A has (0,0),(0,1),(1,0),(1,1)
                        # B has (0,0),(0,1),(1,0),(1,1)
                        # C has (0,0),(0,1),(1,0),(1,1)
                        
                        # Global index mapping
                        a_i0 = offsets[bi]
                        a_i1 = offsets[bi] + 1
                        a_j0 = offsets[bj]
                        a_j1 = offsets[bj] + 1
                        
                        b_j0 = offsets[bj]
                        b_j1 = offsets[bj] + 1
                        b_k0 = offsets[bk]
                        b_k1 = offsets[bk] + 1
                        
                        c_k0 = offsets[bk]
                        c_k1 = offsets[bk] + 1
                        c_i0 = offsets[bi]
                        c_i1 = offsets[bi] + 1
                        
                        # Strassen's 7 products
                        # M1 = (A11+A22)(B11+B22)
                        U1 = {(a_i0, a_j0): 1, (a_i1, a_j1): 1}
                        V1 = {(b_j0, b_k0): 1, (b_j1, b_k1): 1}
                        W1 = {(c_k0, c_i0): 1, (c_k0, c_i1): 1, (c_k1, c_i0): 1, (c_k1, c_i1): 1}
                        # C11 += M1, C22 += M1
                        # Actually: C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
                        # So W1 contributes: C11 += 1, C22 += 1
                        W1 = {(c_k0, c_i0): 1, (c_k1, c_i1): 1}
                        U_terms.append(U1)
                        V_terms.append(V1)
                        W_terms.append(W1)
                        
                        # M2 = (A21+A22)B11
                        U2 = {(a_i1, a_j0): 1, (a_i1, a_j1): 1}
                        V2 = {(b_j0, b_k0): 1}
                        W2 = {(c_k0, c_i1): 1, (c_k1, c_i0): 1}
                        # C21 += M2, C22 -= M2
                        W2 = {(c_k0, c_i1): 1, (c_k1, c_i0): 1, (c_k1, c_i1): -1}
                        U_terms.append(U2)
                        V_terms.append(V2)
                        W_terms.append(W2)
                        
                        # M3 = A11(B21-B22)
                        U3 = {(a_i0, a_j0): 1}
                        V3 = {(b_j1, b_k0): 1, (b_j1, b_k1): -1}
                        W3 = {(c_k0, c_i1): 1, (c_k1, c_i1): 1}
                        # C12 += M3, C22 += M3
                        W3 = {(c_k0, c_i1): 1, (c_k1, c_i1): 1}
                        U_terms.append(U3)
                        V_terms.append(V3)
                        W_terms.append(W3)
                        
                        # M4 = A22(B12-B11)
                        U4 = {(a_i1, a_j1): 1}
                        V4 = {(b_j0, b_k1): 1, (b_j0, b_k0): -1}
                        W4 = {(c_k0, c_i0): 1, (c_k0, c_i1): 1}
                        # C11 += M4, C21 += M4
                        W4 = {(c_k0, c_i0): 1, (c_k0, c_i1): 1}
                        U_terms.append(U4)
                        V_terms.append(V4)
                        W_terms.append(W4)
                        
                        # M5 = (A11+A12)B22
                        U5 = {(a_i0, a_j0): 1, (a_i0, a_j1): 1}
                        V5 = {(b_j1, b_k1): 1}
                        W5 = {(c_k0, c_i1): 1, (c_k1, c_i0): -1}
                        # C12 += M5, C11 -= M5
                        W5 = {(c_k0, c_i1): 1, (c_k0, c_i0): -1}
                        U_terms.append(U5)
                        V_terms.append(V5)
                        W_terms.append(W5)
                        
                        # M6 = (A21-A11)(B11+B12)
                        U6 = {(a_i1, a_j0): 1, (a_i0, a_j0): -1}
                        V6 = {(b_j0, b_k0): 1, (b_j0, b_k1): 1}
                        W6 = {(c_k1, c_i1): 1}
                        # C22 += M6
                        W6 = {(c_k1, c_i1): 1}
                        U_terms.append(U6)
                        V_terms.append(V6)
                        W_terms.append(W6)
                        
                        # M7 = (A12-A22)(B21+B22)
                        U7 = {(a_i0, a_j1): 1, (a_i1, a_j1): -1}
                        V7 = {(b_j1, b_k0): 1, (b_j1, b_k1): 1}
                        W7 = {(c_k0, c_i0): 1}
                        # C11 += M7
                        W7 = {(c_k0, c_i0): 1}
                        U_terms.append(U7)
                        V_terms.append(V7)
                        W_terms.append(W7)
                        
                    else:
                        # Schoolbook for this block pair
                        for ai in range(a_rows):
                            for aj in range(a_cols):
                                for bk in range(b_cols):
                                    global_i = offsets[bi] + ai
                                    global_j = offsets[bj] + aj
                                    global_k = offsets[bk] + bk
                                    
                                    U_term = {(global_i, global_j): 1}
                                    V_term = {(global_j, global_k): 1}
                                    W_term = {(global_k, global_i): 1}
                                    
                                    U_terms.append(U_term)
                                    V_terms.append(V_term)
                                    W_terms.append(W_term)
        
        rank = len(U_terms)
        
        # Build the actual matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for t in range(rank):
            for (i, j), val in U_terms[t].items():
                U[i * m + j][t] = val
            for (j, k), val in V_terms[t].items():
                V[j * p + k][t] = val
            for (k, i), val in W_terms[t].items():
                W[k * n + i][t] = val
        
        return {"U": U, "V": V, "W": W}

    # Try block Strassen first
    try:
        result = block_strassen_decomposition()
        
        # Verify correctness
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        # Check: sum_t U[:,t] V[:,t] W[:,t]^T should give the matrix multiplication tensor
        # The tensor T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        
        # Verify with random test
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((p, n))
        for t in range(len(U)):
            # Extract A_t, B_t, C_t contributions
            a_t = U[:, t].reshape(n, m)
            b_t = V[:, t].reshape(m, p)
            c_t = W[:, t].reshape(p, n)
            # The decomposition: C = sum_t c_t * (a_t : A) * (b_t : B)
            # where (a_t : A) = sum_{i,j} a_t[i,j] * A[i,j]
            a_dot = np.sum(a_t * A_test)
            b_dot = np.sum(b_t * B_test)
            C_computed += c_t * a_dot * b_dot
        
        # Note: C is p x n, but matrix multiplication gives n x p
        # The trace convention: W[k*n+i, r] means C[i,k] = sum ...
        # So C_computed[i,k] should equal C_expected[i,k]
        # But C_computed is indexed as [k,i] from W[k*n+i]
        # Let me re-examine the convention
        
        # W[k*n+i, r] -> this indexes C[i,k] as W[k*n+i, r]
        # So if we compute sum_t W[:,t] * (U[:,t]·A) * (V[:,t]·B), we get a p x n matrix
        # where element [k,i] = C[i,k]
        
        C_computed_transposed = C_computed.T  # Now n x p
        error = np.max(np.abs(C_computed_transposed - C_expected))
        
        if error < 1e-10:
            return result
        else:
            # Verification failed, fall back to schoolbook
            pass
    except Exception:
        pass

    # Fallback to schoolbook
    return schoolbook()
