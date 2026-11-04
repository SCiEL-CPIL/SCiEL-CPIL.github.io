"""Matrix multiplication 5x5x5 - Block decomposition with Strassen for 2x2x2 blocks."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01], [b10, b11]]
    C = A * B
    
    Strassen products:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    C = [[M1+M4-M5+M7, M3+M5],
         [M2+M4,       M1-M2+M3+M6]]
    
    Returns: list of (U_row_indices, U_coeffs, V_row_indices, V_coeffs, W_row_indices, W_coeffs)
    where rows are local 0..3 for A and B (row-major), and for C.
    """
    # A indices: a00=0, a01=1, a10=2, a11=3
    # B indices: b00=0, b01=1, b10=2, b11=3
    # C indices: c00=0, c01=1, c10=2, c11=3
    
    strassen_terms = []
    
    # M1 = (a00+a11)(b00+b11)
    strassen_terms.append((
        [0, 3], [1, 1],      # U: a00 + a11
        [0, 3], [1, 1],      # V: b00 + b11
        [0, 2, 3], [1, 0, 1],  # W: C00 += M1, C11 += M1 (C01, C10 no)
    ))
    
    # M2 = (a10+a11)b00
    strassen_terms.append((
        [2, 3], [1, 1],      # U: a10 + a11
        [0], [1],            # V: b00
        [2, 3], [1, -1],     # W: C10 += M2, C11 -= M2
    ))
    
    # M3 = a00(b01-b11)
    strassen_terms.append((
        [0], [1],            # U: a00
        [1, 3], [1, -1],     # V: b01 - b11
        [1, 3], [1, 1],      # W: C01 += M3, C11 += M3
    ))
    
    # M4 = a11(b10-b00)
    strassen_terms.append((
        [3], [1],            # U: a11
        [2, 0], [1, -1],     # V: b10 - b00
        [0, 2], [1, 1],      # W: C00 += M4, C10 += M4
    ))
    
    # M5 = (a00+a01)b11
    strassen_terms.append((
        [0, 1], [1, 1],      # U: a00 + a01
        [3], [1],            # V: b11
        [0, 1], [-1, 1],     # W: C00 -= M5, C01 += M5
    ))
    
    # M6 = (a10-a00)(b00+b01)
    strassen_terms.append((
        [2, 0], [1, -1],     # U: a10 - a00
        [0, 1], [1, 1],      # V: b00 + b01
        [3], [1],            # W: C11 += M6
    ))
    
    # M7 = (a01-a11)(b10+b11)
    strassen_terms.append((
        [1, 3], [1, -1],     # U: a01 - a11
        [2, 3], [1, 1],      # V: b10 + b11
        [0], [1],            # W: C00 += M7
    ))
    
    return strassen_terms


def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute the tensor: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # The matrix multiplication tensor should be: T[i,j,k] = 1 if (i,j) pairs with (j,k) to produce (k,i)
    # Actually, the multiplication tensor M satisfies: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # In the tensor decomposition framework: (AB)[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c]*A[i,j]) * (sum_l V[j*p+k, c]*B[j,k])
    # Wait, let me reconsider the convention.
    # U has shape (n*m, rank): U[i*m+j, c]
    # V has shape (m*p, rank): V[j*p+k, c]
    # W has shape (p*n, rank): W[k*n+i, c]
    # The product is: (AB)[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[j*p+k, c] * B[j,k])
    # Hmm, that's not quite right either. Let me think again.
    # Standard tensor decomposition for matrix multiplication:
    # (AB)[i,k] = sum_{j=1}^m A[i,j] B[j,k]
    # Tensor form: M[i,j,k] = 1 for all valid (i,j,k), and the multiplication is:
    # (AB)[i,k] = sum_{j,c} U[i,j,c] * V[j,k,c] * W[i,k,c]  -- but W depends on i,k not j
    # 
    # Given the convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # This means: (AB)[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Wait, that doesn't work because the sums over j are separate.
    #
    # Actually the correct interpretation is:
    # (AB)[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # But this would require U and V to be functions that map A and B to scalars via inner products.
    #
    # Let me just verify by checking that the schoolbook decomposition works.
    # For schoolbook: rank = n*m*p, and for each (i,j,k):
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # Then (AB)[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    # = sum_{i',j',k'} W[k*n+i, col(i',j',k')] * (sum_j U[i*m+j, col(i',j',k')]*A[i,j]) * (sum_j V[j*p+k, col(i',j',k')]*B[j,k])
    # For col corresponding to (i',j',k'): W[k*n+i, col] = 1 only if k*n+i == k'*n+i', i.e., i=i' and k=k'
    # U[i*m+j, col] = 1 only if i*m+j == i'*m+j', i.e., i=i' and j=j'
    # V[j*p+k, col] = 1 only if j*p+k == j'*p+k', i.e., j=j' and k=k'
    # So the sum is over (i',j',k') such that i'=i, k'=k, and also j=j' from U and j=j' from V.
    # So only (i,j,k) survives: contribution = 1 * 1 * 1 = 1
    # Sum over j: sum_j A[i,j] * B[j,k] = (AB)[i,k]. Correct!
    
    # So the verification is:
    # For each (i,k), compute sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # and check it equals sum_j A[i,j]*B[j,k]
    
    # Let's test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using decomposition
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            # sum_j U[i*m+j, c] * A[i,j] for each c
            u_vals = np.zeros(rank)
            for j in range(m):
                u_vals += U[i*m+j, :] * A[i, j]
            v_vals = np.zeros(rank)
            for j in range(m):
                v_vals += V[j*p+k, :] * B[j, k]
            C_decomp[i, k] = np.dot(W[k*n+i, :], u_vals * v_vals)
    
    err = np.max(np.abs(C_decomp - C_expected))
    return err < tol, err


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Block decomposition: split 5 into 2+2+1
    # Blocks for A (n x m): rows split [0:2, 2:4, 4:5], cols split [0:2, 2:4, 4:5]
    # Blocks for B (m x p): rows split [0:2, 2:4, 4:5], cols split [0:2, 2:4, 4:5]
    # Blocks for C (n x p): rows split [0:2, 2:4, 4:5], cols split [0:2, 2:4, 4:5]
    
    row_splits_A = [0, 2, 4, 5]  # A row block boundaries
    col_splits_A = [0, 2, 4, 5]  # A col block boundaries (= B row block boundaries)
    row_splits_B = [0, 2, 4, 5]  # B row block boundaries
    col_splits_B = [0, 2, 4, 5]  # B col block boundaries (= C col block boundaries)
    row_splits_C = [0, 2, 4, 5]  # C row block boundaries (= A row block boundaries)
    col_splits_C = [0, 2, 4, 5]  # C col block boundaries
    
    # Block sizes
    A_block_sizes = [(row_splits_A[i+1]-row_splits_A[i], col_splits_A[i+1]-col_splits_A[i]) 
                     for i in range(3) for j in range(3)]
    # Actually, let's index blocks by (i,j) where i is row block, j is col block
    A_block_dims = {}
    for i in range(3):
        for j in range(3):
            hi = row_splits_A[i+1] - row_splits_A[i]
            wj = col_splits_A[j+1] - col_splits_A[j]
            A_block_dims[(i,j)] = (hi, wj)
    
    B_block_dims = {}
    for j in range(3):
        for k in range(3):
            hj = row_splits_B[j+1] - row_splits_B[j]
            wk = col_splits_B[k+1] - col_splits_B[k]
            B_block_dims[(j,k)] = (hj, wk)
    
    C_block_dims = {}
    for i in range(3):
        for k in range(3):
            hi = row_splits_C[i+1] - row_splits_C[i]
            wk = col_splits_C[k+1] - col_splits_C[k]
            C_block_dims[(i,k)] = (hi, wk)
    
    # For each triple of blocks (i,j,k) where A[i,j] * B[j,k] contributes to C[i,k]:
    # The product is a matrix of size A_block_dims[(i,j)][0] x B_block_dims[(j,k)][1]
    # which should equal C_block_dims[(i,k)]
    
    # We'll build U, V, W as lists of lists
    U_rows = n * m  # 25
    V_rows = m * p  # 25
    W_rows = p * n  # 25
    
    # Collect all terms
    terms = []  # list of (u_dict, v_dict, w_dict) where each is {index: coeff}
    
    strassen = _strassen_2x2()
    
    for i in range(3):  # A row block
        for j in range(3):  # A col / B row block
            for k in range(3):  # B col / C col block
                # Get block dimensions
                ha, wa = A_block_dims[(i,j)]
                hb, wb = B_block_dims[(j,k)]
                hc, wc = C_block_dims[(i,k)]
                
                # Check consistency: ha should equal hc, wa should equal hb
                assert ha == hc, f"A rows {ha} != C rows {hc}"
                assert wa == hb, f"A cols {wa} != B rows {hb}"
                
                # Local dimensions
                local_a_rows = ha
                local_a_cols = wa
                local_b_rows = hb
                local_b_cols = wb
                local_c_rows = hc
                local_c_cols = wc
                
                # Global row starts
                a_row_start = row_splits_A[i]
                a_col_start = col_splits_A[j]
                b_row_start = row_splits_B[j]
                b_col_start = col_splits_B[k]
                c_row_start = row_splits_C[i]
                c_col_start = col_splits_C[k]
                
                # If all blocks are 2x2, use Strassen
                if ha == 2 and wa == 2 and hb == 2 and wb == 2:
                    # Use Strassen decomposition
                    for term in strassen:
                        u_local, u_coeffs, v_local, v_coeffs, w_local, w_coeffs = term
                        u_dict = {}
                        for li, coeff in zip(u_local, u_coeffs):
                            if coeff != 0:
                                # Global U index: (a_row_start + li_row) * m + (a_col_start + li_col)
                                # But li is a linear index into 2x2 A block: li = li_row*2 + li_col
                                li_row = li // 2
                                li_col = li % 2
                                global_u = (a_row_start + li_row) * m + (a_col_start + li_col)
                                u_dict[global_u] = u_dict.get(global_u, 0) + coeff
                        
                        v_dict = {}
                        for li, coeff in zip(v_local, v_coeffs):
                            if coeff != 0:
                                li_row = li // 2
                                li_col = li % 2
                                global_v = (b_row_start + li_row) * p + (b_col_start + li_col)
                                v_dict[global_v] = v_dict.get(global_v, 0) + coeff
                        
                        w_dict = {}
                        for li, coeff in zip(w_local, w_coeffs):
                            if coeff != 0:
                                li_row = li // 2
                                li_col = li % 2
                                global_w = (c_col_start + li_col) * n + (c_row_start + li_row)
                                w_dict[global_w] = w_dict.get(global_w, 0) + coeff
                        
                        terms.append((u_dict, v_dict, w_dict))
                else:
                    # Schoolbook for this block
                    for li in range(ha):
                        for lj in range(wa):
                            for lk in range(wb):
                                # A[li, lj] * B[lj, lk] contributes to C[li, lk]
                                global_u = (a_row_start + li) * m + (a_col_start + lj)
                                global_v = (b_row_start + lj) * p + (b_col_start + lk)
                                global_w = (c_col_start + lk) * n + (c_row_start + li)
                                terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    rank = len(terms)
    
    # Build U, V, W
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            if 0 <= idx < U_rows:
                U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            if 0 <= idx < V_rows:
                V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            if 0 <= idx < W_rows:
                W[idx][c] = coeff
    
    # Verify
    valid, err = _verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
