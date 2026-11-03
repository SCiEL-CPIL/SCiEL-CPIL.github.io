import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 4, 6
    
    # Build the rank-66 decomposition
    U66 = [[0] * 66 for _ in range(n * m)]
    V66 = [[0] * 66 for _ in range(m * p)]
    W66 = [[0] * 66 for _ in range(p * n)]
    
    col = 0
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1
        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1
    
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)
    
    # Convert to numpy
    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)
    
    # Vectorized verification
    def verify(U, V, W):
        """Check: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 if j==k else 0"""
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    expected = 1.0 if j == k else 0.0
                    max_err = max(max_err, abs(s - expected))
        return max_err
    
    err66 = verify(U, V, W)
    if err66 > 1e-10:
        return {"U": U66, "V": V66, "W": W66}
    
    # Try to reduce to rank 65 by checking if any column is tensor-redundant
    rank = 66
    best_U, best_V, best_W = U, V, W
    
    # Precompute tensor entries for all columns
    # tensor_entry[i,j,k,l] = U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    tensor = np.zeros((n, m, p, 66))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i,j,k,:] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    
    for c in range(66):
        if time.time() > deadline - 3:
            break
        
        # Check if column c's tensor contribution is in the span of others
        # Build linear system: for each (i,j,k), tensor[i,j,k,c] = sum_{l!=c} alpha_l * tensor[i,j,k,l]
        # Only include non-trivial equations
        eq_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = tensor[i,j,k,c]
                    rhs_nonzero = any(abs(tensor[i,j,k,l]) > 1e-15 for l in range(66) if l != c)
                    if abs(lhs) > 1e-15 or rhs_nonzero:
                        eq_indices.append((i,j,k))
        
        if not eq_indices:
            keep = [l for l in range(66) if l != c]
            U_new = U[:, keep]
            V_new = V[:, keep]
            W_new = W[:, keep]
            if verify(U_new, V_new, W_new) < 1e-10:
                rank = 65
                best_U, best_V, best_W = U_new, V_new, W_new
                break
            continue
        
        # Build matrix A and vector b
        A = np.zeros((len(eq_indices), 65))
        b = np.zeros(len(eq_indices))
        keep_cols = [l for l in range(66) if l != c]
        for idx, (i,j,k) in enumerate(eq_indices):
            b[idx] = tensor[i,j,k,c]
            for col_idx, l in enumerate(keep_cols):
                A[idx, col_idx] = tensor[i,j,k,l]
        
        try:
            alpha, residuals, rnk, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        residual = np.max(np.abs(A @ alpha - b))
        if residual < 1e-10:
            U_new = U[:, keep_cols]
            V_new = V[:, keep_cols]
            W_new = W[:, keep_cols]
            if verify(U_new, V_new, W_new) < 1e-10:
                rank = 65
                best_U, best_V, best_W = U_new, V_new, W_new
                break
    
    # If still rank 66, try a targeted ALS with vectorized updates
    if rank == 66 and time.time() < deadline - 10:
        # Try each column elimination with vectorized ALS
        for c in range(66):
            if time.time() > deadline - 3:
                break
            
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            U_try[:, c] = 0
            V_try[:, c] = 0
            W_try[:, c] = 0
            
            converged = False
            for it in range(20):
                if time.time() > deadline - 3:
                    break
                
                # Update U rows: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # A[k, l] = V[j*p+k, l] * W[k*n+i, l], b[k] = 1 if j==k else 0
                        A_mat = V_try[j*p:(j+1)*p, :] * W_try.reshape(p, n, 1)[:, i, :]
                        b_vec = np.array([1.0 if j == k else 0.0 for k in range(p)])
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            U_try[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V rows: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        A_mat = U_try.reshape(n, m, 66)[:, j, :] * W_try.reshape(p, n, 1)[:, :, k].T
                        b_vec = np.array([1.0 if j == k else 0.0 for i in range(n)])
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            V_try[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W rows: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        A_mat = U_try.reshape(n, m, 66)[i, :, :] * V_try.reshape(m, p, 66)[:, k, :].T
                        b_vec = np.array([1.0 if j == k else 0.0 for j in range(m)])
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            W_try[k*n+i, :] = sol
                        except:
                            pass
                
                err = verify(U_try, V_try, W_try)
                if err < 1e-10:
                    rank = 65
                    best_U, best_V, best_W = U_try, V_try, W_try
                    converged = True
                    break
                if err > 10.0:
                    break
            
            if converged:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
