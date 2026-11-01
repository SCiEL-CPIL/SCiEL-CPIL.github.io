"""
Optimized Matrix Multiplication Tensor Decomposition for 2x4 * 4x4.
Uses Strassen's algorithm on 2x2 blocks to achieve rank 28.
"""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_u][offset + 0] += 1
        U[a11_u][offset + 0] += 1
        V[b00_v][offset + 0] += 1
        V[b11_v][offset + 0] += 1
        W[c00_w][offset + 0] += 1
        W[c11_w][offset + 0] += 1
        
        # M2 = (a10+a11)b00
        U[a10_u][offset + 1] += 1
        U[a11_u][offset + 1] += 1
        V[b00_v][offset + 1] += 1
        W[c10_w][offset + 1] += 1
        W[c11_w][offset + 1] -= 1
        
        # M3 = a00(b01-b11)
        U[a00_u][offset + 2] += 1
        V[b01_v][offset + 2] += 1
        V[b11_v][offset + 2] -= 1
        W[c01_w][offset + 2] += 1
        W[c11_w][offset + 2] += 1
        
        # M4 = a11(b10-b00)
        U[a11_u][offset + 3] += 1
        V[b10_v][offset + 3] += 1
        V[b00_v][offset + 3] -= 1
        W[c00_w][offset + 3] += 1
        W[c10_w][offset + 3] += 1
        
        # M5 = (a00+a01)b11
        U[a00_u][offset + 4] += 1
        U[a01_u][offset + 4] += 1
        V[b11_v][offset + 4] += 1
        W[c00_w][offset + 4] -= 1
        W[c01_w][offset + 4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_u][offset + 5] += 1
        U[a00_u][offset + 5] -= 1
        V[b00_v][offset + 5] += 1
        V[b01_v][offset + 5] += 1
        W[c11_w][offset + 5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_u][offset + 6] += 1
        U[a11_u][offset + 6] -= 1
        V[b10_v][offset + 6] += 1
        V[b11_v][offset + 6] += 1
        W[c00_w][offset + 6] += 1
    
    # Block 1: A_left * B_top_left -> C_left
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_right * B_bottom_left -> C_left
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A_left * B_top_right -> C_right
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A_right * B_bottom_right -> C_right
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    # Attempt to find a linear dependency to reduce rank from 28 to 27
    # Check if any column of the tensor can be expressed as a linear combination of others
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each pair of rank-1 terms, check if one is in the span of the others
    # We look for a linear relation: sum_k lambda_k * (u_k ⊗ v_k ⊗ w_k) = 0
    # This is equivalent to checking if the 28 vectors (flattened rank-1 tensors) are linearly dependent
    
    # Flatten each rank-1 tensor into a vector of length 8*16*8 = 1024
    # But that's expensive. Instead, use a smarter approach.
    
    # Alternative: check if removing one term still gives a valid decomposition
    # by verifying the remaining 27 terms still span the same tensor space
    
    # For a quick check, we can try removing each term and see if the resulting
    # 27-term decomposition still equals the full matrix multiplication tensor.
    
    # The full tensor T[i,j,k] = delta_{i,k} (for matrix mult A*B where A[i,j], B[j,k])
    # Wait, let me reconsider. For A (n x m) * B (m x p) -> C (p x n) with trace convention W[k*n+i, r]
    # Actually the tensor is T[a, b, c] where a indexes A, b indexes B, c indexes C
    # T[a, b, c] = 1 if the multiplication contributes to that output element
    
    # Let me build the full multiplication tensor and check rank
    # T[i, j, k] where i in [0, n*m), j in [0, m*p), k in [0, p*n)
    # For standard matrix mult: C = A * B, but here we have trace convention
    
    # The tensor for 2x4 * 4x4 multiplication:
    # A has 8 elements, B has 16 elements, C has 8 elements
    # T[a, b, c] = 1 if A[a] * B[b] contributes to C[c]
    
    # Let me construct this properly
    # A is 2x4: A[row, col] for row in [0,2), col in [0,4)
    # B is 4x4: B[row, col] for row in [0,4), col in [0,4)
    # C is 4x2: C[row, col] for row in [0,4), col in [0,2)
    # C = A * B: C[i, j] = sum_k A[i, k] * B[k, j]
    # But with trace convention W[k*n+i, r], so C is indexed as C[row, col] -> index row*n + col
    
    # Wait, the problem says "output factor follows the official trace convention W[k*n+i,r]"
    # This suggests C is a 4x2 matrix indexed as W[k*n+i] where k is column, i is row
    # So C[col, row] = W[col*2 + row]
    
    # For matrix mult A (2x4) * B (4x4) -> C (4x2):
    # C[j, i] = sum_k A[i, k] * B[k, j]
    # A[i, k] has index i*4 + k
    # B[k, j] has index k*4 + j
    # C[j, i] has index j*2 + i
    
    # So T[a, b, c] = 1 if there exists i, k, j such that:
    # a = i*4 + k, b = k*4 + j, c = j*2 + i
    # i in [0,2), k in [0,4), j in [0,4)
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1
    
    # Now check if we can remove one of the 28 rank-1 terms
    # and still have the sum equal T
    
    def check_decomposition(U_mat, V_mat, W_mat):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T"""
        # Compute the tensor from the decomposition
        result = np.zeros((n*m, m*p, p*n))
        for r in range(U_mat.shape[1]):
            u = U_mat[:, r]
            v = V_mat[:, r]
            w = W_mat[:, r]
            result += np.einsum('a,b,c->abc', u, v, w)
        return np.allclose(result, T, atol=1e-10)
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = 28
    
    # Try removing each of the 28 terms
    for remove_idx in range(28):
        if time.time() > deadline - 5:
            break
        
        # Create copies with the remove_idx column zeroed out
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        for i in range(n*m):
            U_test[i][remove_idx] = 0
        for i in range(m*p):
            V_test[i][remove_idx] = 0
        for i in range(p*n):
            W_test[i][remove_idx] = 0
        
        U_np_test = np.array(U_test, dtype=float)
        V_np_test = np.array(V_test, dtype=float)
        W_np_test = np.array(W_test, dtype=float)
        
        if check_decomposition(U_np_test, V_np_test, W_np_test):
            best_rank = 27
            best_result = {"U": U_test, "V": V_test, "W": W_test}
            break
    
    # If we didn't find a simple removal, try a more sophisticated approach
    # Look for a linear combination of terms that sums to zero
    if best_rank == 28 and time.time() < deadline - 10:
        # Build the 28 rank-1 tensor vectors
        vecs = []
        for r in range(28):
            u = np.array(U[:, r], dtype=float)
            v = np.array(V[:, r], dtype=float)
            w = np.array(W[:, r], dtype=float)
            vec = np.einsum('a,b,c->abc', u, v, w).flatten()
            vecs.append(vec)
        
        vecs = np.array(vecs)  # shape (28, 1024)
        
        # Find null space of vecs.T
        # If there's a non-trivial null vector, we have a linear dependency
        try:
            _, s, vh = np.linalg.svd(vecs.T, full_matrices=False)
            # The smallest singular value corresponds to the null direction
            # If s[-1] is very small, we have a dependency
            if s[-1] < 1e-10:
                # Find the null vector
                null_vec = vh[-1]
                # Identify which terms are involved
                involved = [i for i in range(28) if abs(null_vec[i]) > 1e-10]
                
                # Try to eliminate the term with the largest coefficient
                if involved:
                    max_idx = max(involved, key=lambda i: abs(null_vec[i]))
                    # Create a modified decomposition where we redistribute the contribution
                    # This is complex; for now, just note the dependency
                    pass
        except:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
