import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # safety margin before 150s kill
    
    # Try to find a rank-38 decomposition by modifying the rank-39 structure
    # Key idea: Can we merge one of the Strassen terms with the schoolbook terms?
    
    # First, let's try a rank-38 approach:
    # Use 3 Strassen blocks (21 terms) + 3 schoolbook terms for last column + 14 for third row = 38?
    # But that doesn't work because we need 4 terms for the last column.
    
    # Alternative: Can we use a different decomposition for the 2x3 x 3x7 case?
    # Let's try to use a tensor decomposition that shares more across columns.
    
    # For now, keep the known rank-39 decomposition as fallback
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][col] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][col] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][col] += coeff
        col += 1
    
    def add_strassen_block(k_start):
        a00, a01, a10, a11 = 0, 1, 3, 4
        b00 = 0 * p + k_start
        b01 = 0 * p + k_start + 1
        b10 = 1 * p + k_start
        b11 = 1 * p + k_start + 1
        c00 = k_start * n + 0
        c01 = (k_start + 1) * n + 0
        c10 = k_start * n + 1
        c11 = (k_start + 1) * n + 1
        
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
        add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
        add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])
    
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    b06 = 0 * p + 6
    b16 = 1 * p + 6
    c06 = 6 * n + 0
    c16 = 6 * n + 1
    
    add_term([0], [1], [b06], [1], [c06], [1])
    add_term([1], [1], [b16], [1], [c06], [1])
    add_term([3], [1], [b06], [1], [c16], [1])
    add_term([4], [1], [b16], [1], [c16], [1])
    
    for k in range(p):
        b2k = 2 * p + k
        c0k = k * n + 0
        c1k = k * n + 1
        add_term([2], [1], [b2k], [1], [c0k], [1])
        add_term([5], [1], [b2k], [1], [c1k], [1])
    
    # Verify the decomposition is correct
    def verify_decomposition(U_mat, V_mat, W_mat, rank_val):
        """Verify that the decomposition correctly computes matrix multiplication"""
        # Convert to numpy arrays
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        # Check shapes
        if U_arr.shape != (n * m, rank_val):
            return False
        if V_arr.shape != (m * p, rank_val):
            return False
        if W_arr.shape != (p * n, rank_val):
            return False
        
        # For each term r, we have U[:,r], V[:,r], W[:,r]
        # The decomposition should satisfy:
        # sum_r U[i,r] * V[j,r] = W[k,r] where i,j,k are properly mapped
        
        # Let's verify by checking a few random matrices
        import random
        random.seed(42)
        
        for _ in range(5):
            A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
            
            # Compute A*B normally
            C_expected = [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
            
            # Compute using decomposition
            # C[i,j] = sum_r U[i*m+j, r] * V[?, r] * W[?, r]
            # Actually, the decomposition is:
            # C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_l V[l*p+j, r] * B[l,j]) * W[j*n+i, r]
            
            C_decomp = [[0.0 for _ in range(p)] for _ in range(n)]
            for r in range(rank_val):
                for i in range(n):
                    for j in range(p):
                        # u_part = sum_k A[i,k] * U[i*m+k, r]
                        u_part = sum(A[i][k] * U_arr[i*m+k, r] for k in range(m))
                        # v_part = sum_l B[l,j] * V[l*p+j, r]
                        v_part = sum(B[l][j] * V_arr[l*p+j, r] for l in range(m))
                        # w_part = W[j*n+i, r]
                        w_part = W_arr[j*n+i, r]
                        C_decomp[i][j] += u_part * v_part * w_part
            
            # Check if C_decomp matches C_expected
            for i in range(n):
                for j in range(p):
                    if abs(C_decomp[i][j] - C_expected[i][j]) > 1e-6:
                        return False
        
        return True
    
    # Verify our decomposition
    if not verify_decomposition(U, V, W, rank):
        # If verification fails, we have a bug
        raise ValueError("Decomposition verification failed")
    
    if time.time() - start_time > time_limit:
        raise TimeoutError("Exceeded time limit")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
