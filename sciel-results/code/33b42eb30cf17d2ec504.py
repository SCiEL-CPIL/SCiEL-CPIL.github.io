import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Build Strassen 2x2x2 blocks
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Stacked matrix for dependency check
    # Rows are elements of U, V, W. Cols are the rank-1 terms.
    stacked = np.vstack([U_np, V_np, W_np])
    
    # Perform SVD to find rank
    try:
        U_svd, s, Vt = np.linalg.svd(stacked, full_matrices=False)
        tol = 1e-10
        r = np.sum(s > tol)
        
        if r < rank:
            # We have dependencies.
            # We want to find a set of r independent columns.
            # The right singular vectors corresponding to non-zero singular values span the row space.
            # The columns of 'stacked' are linear combinations of these.
            
            # A safer way: Gaussian elimination on the stacked matrix to find pivot columns.
            # But SVD gives us the rank.
            
            # Let's try to eliminate columns one by one.
            # If column c is in the span of others, we can drop it.
            
            # We can use the null space of the stacked matrix.
            # If v is in null space, sum(v_i * col_i) = 0.
            # Then col_j = -1/v_j * sum_{i!=j} v_i * col_i.
            # We can drop column j.
            
            null_space = Vt[r:]  # shape (rank-r, rank)
            
            # We will greedily remove columns.
            # Start with all columns.
            active_cols = list(range(rank))
            
            # We need to be careful: removing a column changes the dependencies.
            # However, if we have a null vector with a non-zero entry at j, we can eliminate j.
            
            # Let's just take the first r columns?
            # This is valid if the first r columns are independent.
            # SVD doesn't guarantee the first r columns of the ORIGINAL matrix are independent.
            
            # Better: Use QR decomposition of the stacked matrix.
            # The pivot columns of QR are independent.
            Q, R = np.linalg.qr(stacked, mode='reduced')
            # The diagonal of R indicates rank.
            # The pivot columns are those with large diagonal entries.
            
            # Actually, simply taking the first r columns often works for structured matrices like Strassen.
            # Let's verify if the first r columns form a valid decomposition.
            
            U_reduced = U_np[:, :r]
            V_reduced = V_np[:, :r]
            W_reduced = W_np[:, :r]
            
            # Verify
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p))
            for c in range(r):
                # C += (U[:,c] . A) * (V[:,c] . B) * W[:,c].reshape(n,p)
                # Wait, U is (n*m, 1), A is (n, m).
                # dot(U[:,c], A.flatten())
                sa = np.dot(U_reduced[:, c], A_test.flatten())
                sb = np.dot(V_reduced[:, c], B_test.flatten())
                C_computed += sa * sb * W_reduced[:, c].reshape(n, p)
            
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
            
            # If it failed, try to find a better set of columns.
            # Use the pivot indices from QR.
            # np.linalg.qr doesn't return pivots directly in standard numpy.
            # We can use scipy.linalg.qr with pivoting?
            # scipy is available.
            
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
