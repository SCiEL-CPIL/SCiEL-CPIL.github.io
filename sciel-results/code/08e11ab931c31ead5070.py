"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses Strassen's algorithm on 2x2 sub-blocks to reduce rank from 50 to 46.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_strassen_decomposition():
    """
    Build the rank-46 Strassen-based decomposition for 2x5 x 5x5.
    Partition inner dimension 5 = 2+2+1.
    """
    n, m, p = 2, 5, 5

    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        # M1 = (a+d)(e+h)
        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        # M2 = (c+d)e
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        # M3 = a(f-h)
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        # M4 = d(g-e)
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        # M5 = (a+b)h
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        # M6 = (c-a)(e+f)
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        # M7 = (b-d)(g+h)
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    # Partition 5 = 2+2+1
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)

    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)

    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)

    total_rank = len(terms)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val

    return U, V, W, total_rank


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = 2, 5, 5
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Compute the tensor from the decomposition
    # T[i,j,k] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    # Actually, the convention is:
    # (A*B)[i,j] = sum_k A[i,k] * B[k,j]
    # Decomposition: (A*B)[i,j] = sum_r (sum_k A[i,k]*U[k,i,r]) * (sum_k B[k,j]*V[k,j,r]) * W[j,i,r]
    # Wait, let me re-check the convention.
    # U has shape (n*m, rank), so U[i*m + k, r] corresponds to A[i,k]
    # V has shape (m*p, rank), so V[k*p + j, r] corresponds to B[k,j]
    # W has shape (p*n, rank), so W[j*n + i, r] corresponds to (A*B)[i,j]
    
    # So: (A*B)[i,j] = sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j])
    
    # Let's test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        U_col = U_np[:, r].reshape(n, m)
        V_col = V_np[:, r].reshape(m, p)
        W_col = W_np[:, r].reshape(p, n)
        
        # Compute the rank-1 contribution
        left = A @ U_col  # n x m @ m x m? No, U_col is n x m
        # Actually: sum_k A[i,k] * U[i*m+k, r] = A @ U_col (where U_col[i,k] = U[i*m+k, r])
        # This is just A @ U_col if U_col is shaped as (n, m)
        # But U_col[i, k] = U[i*m + k, r], which is exactly the reshape
        left = A @ U_col  # shape (n, m) @ (n, m)? No, that's wrong.
        
        # Let me re-think. U has entries U[i*m + k, r].
        # The contraction is sum_k A[i,k] * U[i*m+k, r].
        # If we reshape U[:, r] to (n, m), then U_reshaped[i, k] = U[i*m + k, r].
        # So sum_k A[i,k] * U_reshaped[i, k] is the element-wise product summed over k.
        # This is NOT A @ U_reshaped. It's the Frobenius inner product of A and U_reshaped.
        
        left = np.sum(A * U_col)  # scalar
        right = np.sum(B * V_col)  # scalar
        C_computed += W_col.T * (left * right)
        # W_col is (p, n), so W_col.T is (n, p), and W_col.T[i, j] = W[j*n + i, r]
    
    max_err = np.max(np.abs(C_computed - C_expected))
    return max_err


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the standard rank-46 decomposition
    U, V, W, total_rank = _build_strassen_decomposition()

    # Verify the decomposition
    max_err = _verify_decomposition(U, V, W)
    if max_err > 1e-10:
        raise ValueError(f"Decomposition verification failed with max error {max_err}")

    # Time guard
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
