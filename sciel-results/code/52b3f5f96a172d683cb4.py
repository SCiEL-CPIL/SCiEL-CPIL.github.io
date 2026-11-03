import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    b = 3
    rank_3x3 = b * b * b  # 27
    total_rank = 7 * rank_3x3  # 189
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Strassen terms: (a_coeffs, b_coeffs, c_contribs)
    # a_coeffs: {(bi,bj): coeff} for A block (bi,bj)
    # b_coeffs: {(bi,bj): coeff} for B block (bi,bj)
    # c_contribs: [(ci,cj, coeff)] for C block (ci,cj)
    terms = [
        # M1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            [((0,0), 1), ((1,1), 1)]
        ),
        # M2: (A10+A11)B00 -> C10(+1), C11(-1)
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            [((1,0), 1), ((1,1), -1)]
        ),
        # M3: A00(B01-B11) -> C01(+1), C11(+1)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            [((0,1), 1), ((1,1), 1)]
        ),
        # M4: A11(B10-B00) -> C00(+1), C10(+1)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            [((0,0), 1), ((1,0), 1)]
        ),
        # M5: (A00+A01)B11 -> C00(-1), C01(+1)
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            [((0,0), -1), ((0,1), 1)]
        ),
        # M6: (A10-A00)(B00+B01) -> C11(+1)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            [((1,1), 1)]
        ),
        # M7: (A01-A11)(B10+B11) -> C00(+1)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            [((0,0), 1)]
        ),
    ]
    
    for t_idx, (a_coeffs, b_coeffs, c_contribs) in enumerate(terms):
        base_rank = t_idx * rank_3x3
        
        for i in range(b):
            for j in range(b):
                for k in range(b):
                    r_local = i * b * b + j * b + k
                    r_global = base_rank + r_local
                    
                    # U: row = (bi*b + i)*m + (bj*b + j)
                    for (bi, bj), coeff_a in a_coeffs.items():
                        row_u = (bi * b + i) * m + (bj * b + j)
                        U[row_u, r_global] += coeff_a
                    
                    # V: row = (bi*b + j)*p + (bj*b + k)
                    for (bi, bj), coeff_b in b_coeffs.items():
                        row_v = (bi * b + j) * p + (bj * b + k)
                        V[row_v, r_global] += coeff_b
                    
                    # W: row = (cj*b + k)*n + (ci*b + i)
                    for (ci, cj), coeff_c in c_contribs:
                        row_w = (cj * b + k) * n + (ci * b + i)
                        W[row_w, r_global] += coeff_c
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        try:
            np.random.seed(42)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # C[i,j] = sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j])
            # Vectorized: 
            # For each r: u_r = U[:,r].reshape(n,m), v_r = V[:,r].reshape(m,p), w_r = W[:,r].reshape(p,n)
            # C_r[i,j] = w_r[j,i] * (A @ ... ) 
            # Actually: sum_k U[i*m+k,r]*A[i,k] = (A * U[:,r].reshape(n,m)).sum(axis=1)[i]
            # sum_k V[k*p+j,r]*B[k,j] = (B * V[:,r].reshape(m,p)).sum(axis=0)[j]
            
            C_computed = np.zeros((n, p), dtype=float)
            for r in range(total_rank):
                u_mat = U[:, r].reshape(n, m)
                v_mat = V[:, r].reshape(m, p)
                w_mat = W[:, r].reshape(p, n)
                
                # a_r[i] = sum_k U[i*m+k,r] * A[i,k]
                a_r = (A * u_mat).sum(axis=1)  # shape (n,)
                # b_r[j] = sum_k V[k*p+j,r] * B[k,j]
                b_r = (B * v_mat).sum(axis=0)  # shape (p,)
                # C_r[i,j] = w_mat[j,i] * a_r[i] * b_r[j]
                C_computed += np.outer(a_r, b_r) * w_mat.T
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Fallback: schoolbook decomposition (rank 216)
                U = np.zeros((n * m, n*m*p), dtype=float)
                V = np.zeros((m * p, n*m*p), dtype=float)
                W = np.zeros((p * n, n*m*p), dtype=float)
                idx = 0
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            U[i*m+k, idx] = 1.0
                            V[k*p+j, idx] = 1.0
                            W[j*n+i, idx] = 1.0
                            idx += 1
                total_rank = n*m*p
        except Exception:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
