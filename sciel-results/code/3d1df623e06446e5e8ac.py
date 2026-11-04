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
    """
    strassen_terms = []
    
    # M1 = (a00+a11)(b00+b11)
    strassen_terms.append((
        [0, 3], [1, 1],
        [0, 3], [1, 1],
        [0, 2, 3], [1, 0, 1],
    ))
    
    # M2 = (a10+a11)b00
    strassen_terms.append((
        [2, 3], [1, 1],
        [0], [1],
        [2, 3], [1, -1],
    ))
    
    # M3 = a00(b01-b11)
    strassen_terms.append((
        [0], [1],
        [1, 3], [1, -1],
        [1, 3], [1, 1],
    ))
    
    # M4 = a11(b10-b00)
    strassen_terms.append((
        [3], [1],
        [2, 0], [1, -1],
        [0, 2], [1, 1],
    ))
    
    # M5 = (a00+a01)b11
    strassen_terms.append((
        [0, 1], [1, 1],
        [3], [1],
        [0, 1], [-1, 1],
    ))
    
    # M6 = (a10-a00)(b00+b01)
    strassen_terms.append((
        [2, 0], [1, -1],
        [0, 1], [1, 1],
        [3], [1],
    ))
    
    # M7 = (a01-a11)(b10+b11)
    strassen_terms.append((
        [1, 3], [1, -1],
        [2, 3], [1, 1],
        [0], [1],
    ))
    
    return strassen_terms

def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-10):
    """Verify that the tensor decomposition is exact using vectorized operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Vectorized computation
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For each (i,k): C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # Reshape for vectorized ops
    U_reshaped = U.reshape(n, m, rank)  # U[i, j, r]
    V_reshaped = V.reshape(m, p, rank)  # V[j, k, r]
    W_reshaped = W.reshape(p, n, rank)  # W[k, i, r]
    
    # Compute u_vals[i, r] = sum_j U[i, j, r] * A[i, j]
    u_vals = np.einsum('ijr,ij->ir', U_reshaped, A)  # shape (n, rank)
    
    # Compute v_vals[k, r] = sum_j V[j, k, r] * B[j, k]
    v_vals = np.einsum('jkr,jk->kr', V_reshaped, B)  # shape (p, rank)
    
    # C[i, k] = sum_r W[k, i, r] * u_vals[i, r] * v_vals[k, r]
    C_decomp = np.einsum('kir,ir,kr->ik', W_reshaped, u_vals, v_vals)
    
    err = np.max(np.abs(C_decomp - C_expected))
    return err < tol, err

def _winograd_3x3():
    """
    Return the 16-term Winograd decomposition for 3x3 matrix multiplication.
    
    A = [[a00, a01, a02],
         [a10, a11, a12],
         [a20, a21, a22]]
    B = [[b00, b01, b02],
         [b10, b11, b12],
         [b20, b21, b22]]
    
    Winograd F2T2:
    F2T2 = [[1, 0], [1, 1], [1, -1]]
    F2T2_inv = [[1, 1/2, -1/2], [0, 1/2, 1/2]]
    
    For 3x3:
    a' = F2T2 * A * F2T2^T (6x6)
    b' = F2T2 * B * F2T2^T (6x6)
    But we only need 16 products from the 6x6.
    
    Actually, the standard Winograd 3x3 uses:
    m = [[(a00+a10+a20), (b00+b10+b20)],
         [(a01+a11+a21), (b01+b11+b21)],
         ...
    ]
    
    Let me use the explicit 16-term decomposition:
    """
    # Winograd 3x3x3 decomposition with 16 terms
    # Using the standard formulation
    terms = []
    
    # The 3x3 Winograd uses transforms:
    # For A: rows grouped as (0,2) and col 1 separate, etc.
    # Actually, let me use a known 16-term decomposition
    
    # Standard Winograd 3x3:
    # a0 = a00 + a20
    # a1 = a01 + a21  
    # a2 = a10
    # a3 = a11
    # a4 = a12
    # a5 = a02 + a22
    # a6 = a00
    # a7 = a01
    # a8 = a02
    # a9 = a20
    # a10 = a21
    # a11 = a22
    
    # b0 = b00 + b02
    # b1 = b10 + b12
    # b2 = b01
    # b3 = b11
    # b4 = b21
    # b5 = b20 + b22
    # b6 = b00
    # b7 = b01
    # b8 = b02
    # b9 = b20
    # b10 = b21
    # b11 = b22
    
    # m0 = a0 * b0
    # m1 = a1 * b1
    # m2 = a2 * b2
    # m3 = a3 * b3
    # m4 = a4 * b4
    # m5 = a5 * b5
    # m6 = a6 * b6
    # m7 = a7 * b7
    # m8 = a8 * b8
    # m9 = a9 * b9
    # m10 = a10 * b10
    # m11 = a11 * b11
    # m12 = (a6+a9) * (b6+b9)  -- wait, this isn't quite right
    
    # Let me use a cleaner formulation. The Winograd 3x3x3 has rank 16.
    # I'll use the decomposition from the literature.
    
    # Actually, let me just implement it directly with the transform matrices.
    # F2T2 = [[1,0],[1,1],[1,-1]]
    # The 3x3 Winograd decomposes into:
    # - One 2x2x2 Strassen-like part (but not exactly)
    # - Additional terms
    
    # For simplicity and correctness, let me use a verified 16-term decomposition.
    # Reference: Winograd, "On the number of multiplications required to compute
    #           the product of two n-by-n matrices" (1968)
    
    # Here's a working 16-term decomposition for 3x3x3:
    
    # Transform A:
    # a00' = a00 + a10 + a20
    # a01' = a01 + a21
    # a02' = a02 + a22
    # a10' = a10
    # a11' = a11
    # a12' = a12
    # a20' = a00
    # a21' = a01
    # a22' = a02
    # a30' = a20
    # a31' = a21
    # a32' = a22
    
    # This is getting complex. Let me use a different approach:
    # Build the decomposition from the Kronecker product structure.
    
    # For now, I'll return None to indicate Winograd isn't implemented,
    # and the code will fall back to schoolbook for 3x3x3.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Partition: 2 + 2 + 1
    row_splits_A = [0, 2, 4, 5]
    col_splits_A = [0, 2, 4, 5]
    row_splits_B = [0, 2, 4, 5]
    col_splits_B = [0, 2, 4, 5]
    row_splits_C = [0, 2, 4, 5]
    col_splits_C = [0, 2, 4, 5]
    
    num_blocks = 3
    
    A_block_dims = {}
    for i in range(num_blocks):
        for j in range(num_blocks):
            hi = row_splits_A[i+1] - row_splits_A[i]
            wj = col_splits_A[j+1] - col_splits_A[j]
            A_block_dims[(i,j)] = (hi, wj)
            
    B_block_dims = {}
    for j in range(num_blocks):
        for k in range(num_blocks):
            hj = row_splits_B[j+1] - row_splits_B[j]
            wk = col_splits_B[k+1] - col_splits_B[k]
            B_block_dims[(j,k)] = (hj, wk)
            
    C_block_dims = {}
    for i in range(num_blocks):
        for k in range(num_blocks):
            hi = row_splits_C[i+1] - row_splits_C[i]
            wk = col_splits_C[k+1] - col_splits_C[k]
            C_block_dims[(i,k)] = (hi, wk)
            
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    terms = []
    
    strassen = _strassen_2x2()
    
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                ha, wa = A_block_dims[(i,j)]
                hb, wb = B_block_dims[(j,k)]
                hc, wc = C_block_dims[(i,k)]
                
                assert ha == hc
                assert wa == hb
                
                a_row_start = row_splits_A[i]
                a_col_start = col_splits_A[j]
                b_row_start = row_splits_B[j]
                b_col_start = col_splits_B[k]
                c_row_start = row_splits_C[i]
                c_col_start = col_splits_C[k]
                
                if ha == 2 and wa == 2 and hb == 2 and wb == 2:
                    # Strassen 2x2x2
                    for term in strassen:
                        u_local, u_coeffs, v_local, v_coeffs, w_local, w_coeffs = term
                        u_dict = {}
                        for li, coeff in zip(u_local, u_coeffs):
                            if coeff != 0:
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
                    # Schoolbook for non-2x2x2 blocks
                    for li in range(ha):
                        for lj in range(wa):
                            for lk in range(wb):
                                global_u = (a_row_start + li) * m + (a_col_start + lj)
                                global_v = (b_row_start + lj) * p + (b_col_start + lk)
                                global_w = (c_col_start + lk) * n + (c_row_start + li)
                                terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    # Deduplicate terms: merge terms with identical (u_dict, v_dict, w_dict)
    # by combining their coefficients
    from collections import defaultdict
    merged = defaultdict(lambda: ({}, {}, {}))
    for u_dict, v_dict, w_dict in terms:
        key = (tuple(sorted(u_dict.items())), tuple(sorted(v_dict.items())), tuple(sorted(w_dict.items())))
        merged[key][0].update({k: v for k, v in u_dict.items()})
        merged[key][1].update({k: v for k, v in v_dict.items()})
        merged[key][2].update({k: v for k, v in w_dict.items()})
    
    # Actually, deduplication won't help here since each term is unique.
    # Keep original terms.
    # terms = [tuple(v) for v in merged.values()]
    
    rank = len(terms)
    
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
                
    valid, err = _verify_decomposition(U, V, W)
    
    if not valid:
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
