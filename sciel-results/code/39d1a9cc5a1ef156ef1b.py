import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 4, 6
    
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
    
    # Build the rank-66 decomposition (baseline)
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
    
    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)
    
    err66 = verify(U, V, W)
    if err66 > 1e-10:
        return {"U": U66, "V": V66, "W": W66}
    
    rank = 66
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce to rank 65 by checking if any column is tensor-redundant
    tensor = np.zeros((n, m, p, 66))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i,j,k,:] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    
    for c in range(66):
        if time.time() > deadline - 3:
            break
        
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
                
                for i in range(n):
                    for j in range(m):
                        A_mat = V_try[j*p:(j+1)*p, :] * W_try.reshape(p, n, 1)[:, i, :]
                        b_vec = np.array([1.0 if j == k else 0.0 for k in range(p)])
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            U_try[i*m+j, :] = sol
                        except:
                            pass
                
                for j in range(m):
                    for k in range(p):
                        A_mat = U_try.reshape(n, m, 66)[:, j, :] * W_try.reshape(p, n, 1)[:, :, k].T
                        b_vec = np.array([1.0 if j == k else 0.0 for i in range(n)])
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            V_try[j*p+k, :] = sol
                        except:
                            pass
                
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
    
    # Explore rank-64: 2+2 B-row splitting
    if time.time() < deadline - 15:
        # Build rank-64 candidate using 2+2 B-row splits
        # A is 3x4, B is 4x6, C is 3x6
        # Split A rows: 1+2, A cols: 2+2
        # Split B rows: 2+2, B cols: 2+2
        # Split C rows: 1+2, C cols: 2+2
        
        # Block sizes: A: (1,2)x(2,2), B: (2,2)x(2,2), C: (1,2)x(2,2)
        # 3x3 block grid, each block uses Strassen-like or direct
        
        U64 = [[0] * 64 for _ in range(n * m)]
        V64 = [[0] * 64 for _ in range(m * p)]
        W64 = [[0] * 64 for _ in range(p * n)]
        
        col64 = 0
        def add_block_1x2_2x2_1x2(a_r, a_c, b_r, b_c, c_r, c_c):
            """A block: 1x2, B block: 2x2, C block: 1x2 (rank 4)"""
            nonlocal col64
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r
            
            U64[A0][col64] += 1; V64[B00][col64] += 1; W64[C0][col64] += 1
            col64 += 1
            U64[A1][col64] += 1; V64[B10][col64] += 1; W64[C0][col64] += 1
            col64 += 1
            U64[A0][col64] += 1; V64[B01][col64] += 1; W64[C1][col64] += 1
            col64 += 1
            U64[A1][col64] += 1; V64[B11][col64] += 1; W64[C1][col64] += 1
            col64 += 1
        
        def add_block_2x2_2x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            """A block: 2x2, B block: 2x2, C block: 2x2 (rank 7 Strassen)"""
            nonlocal col64
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
            
            U64[A00][col64] += 1; U64[A11][col64] += 1
            V64[B00][col64] += 1; V64[B11][col64] += 1
            W64[C00][col64] += 1; W64[C11][col64] += 1
            col64 += 1
            U64[A10][col64] += 1; U64[A11][col64] += 1
            V64[B00][col64] += 1
            W64[C10][col64] += 1; W64[C11][col64] -= 1
            col64 += 1
            U64[A00][col64] += 1
            V64[B01][col64] += 1; V64[B11][col64] -= 1
            W64[C01][col64] += 1; W64[C11][col64] += 1
            col64 += 1
            U64[A11][col64] += 1
            V64[B10][col64] += 1; V64[B00][col64] -= 1
            W64[C00][col64] += 1; W64[C10][col64] += 1
            col64 += 1
            U64[A00][col64] += 1; U64[A01][col64] += 1
            V64[B11][col64] += 1
            W64[C00][col64] -= 1; W64[C01][col64] += 1
            col64 += 1
            U64[A10][col64] += 1; U64[A00][col64] -= 1
            V64[B00][col64] += 1; V64[B01][col64] += 1
            W64[C11][col64] += 1
            col64 += 1
            U64[A01][col64] += 1; U64[A11][col64] -= 1
            V64[B10][col64] += 1; V64[B11][col64] += 1
            W64[C00][col64] += 1
            col64 += 1
        
        def add_block_1x2_2x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            """A block: 1x2, B block: 2x2, C block: 2x2 (rank 4)"""
            nonlocal col64
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)
            
            U64[A0][col64] += 1; V64[B00][col64] += 1; W64[C00][col64] += 1
            col64 += 1
            U64[A1][col64] += 1; V64[B10][col64] += 1; W64[C00][col64] += 1
            col64 += 1
            U64[A0][col64] += 1; V64[B01][col64] += 1; W64[C10][col64] += 1
            col64 += 1
            U64[A1][col64] += 1; V64[B11][col64] += 1; W64[C10][col64] += 1
            col64 += 1
        
        # A rows: 1+2, A cols: 2+2
        # B rows: 2+2, B cols: 2+2
        # C rows: 1+2, C cols: 2+2
        
        # Block (0,0): A[0:1,0:2] x B[0:2,0:2] -> C[0:1,0:2]
        add_block_1x2_2x2_1x2(0, 0, 0, 0, 0, 0)
        # Block (0,1): A[0:1,2:4] x B[0:2,2:4] -> C[0:1,2:4]
        add_block_1x2_2x2_1x2(0, 2, 0, 2, 0, 2)
        # Block (0,2): A[0:1,0:2] x B[0:2,4:6] -> C[0:1,4:6]
        add_block_1x2_2x2_1x2(0, 0, 0, 4, 0, 4)
        # Block (0,3): A[0:1,2:4] x B[0:2,4:6] -> C[0:1,4:6]
        add_block_1x2_2x2_1x2(0, 2, 0, 4, 0, 4)
        
        # Block (1,0): A[1:3,0:2] x B[0:2,0:2] -> C[1:3,0:2]
        add_block_2x2_2x2_2x2(1, 0, 0, 0, 1, 0)
        # Block (1,1): A[1:3,2:4] x B[0:2,2:4] -> C[1:3,2:4]
        add_block_2x2_2x2_2x2(1, 2, 0, 2, 1, 2)
        # Block (1,2): A[1:3,0:2] x B[0:2,4:6] -> C[1:3,4:6]
        add_block_2x2_2x2_2x2(1, 0, 0, 4, 1, 4)
        # Block (1,3): A[1:3,2:4] x B[0:2,4:6] -> C[1:3,4:6]
        add_block_2x2_2x2_2x2(1, 2, 0, 4, 1, 4)
        
        # Block (2,0): A[1:3,0:2] x B[2:4,0:2] -> C[1:3,0:2]
        add_block_2x2_2x2_2x2(1, 0, 2, 0, 1, 0)
        # Block (2,1): A[1:3,2:4] x B[2:4,2:4] -> C[1:3,2:4]
        add_block_2x2_2x2_2x2(1, 2, 2, 2, 1, 2)
        # Block (2,2): A[1:3,0:2] x B[2:4,4:6] -> C[1:3,4:6]
        add_block_2x2_2x2_2x2(1, 0, 2, 4, 1, 4)
        # Block (2,3): A[1:3,2:4] x B[2:4,4:6] -> C[1:3,4:6]
        add_block_2x2_2x2_2x2(1, 2, 2, 4, 1, 4)
        
        U64_arr = np.array(U64, dtype=float)
        V64_arr = np.array(V64, dtype=float)
        W64_arr = np.array(W64, dtype=float)
        
        err64 = verify(U64_arr, V64_arr, W64_arr)
        if err64 < 1e-10:
            rank = 64
            best_U, best_V, best_W = U64_arr, V64_arr, W64_arr
            # Try to reduce to rank 63
            if time.time() < deadline - 5:
                tensor64 = np.zeros((n, m, p, 64))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            tensor64[i,j,k,:] = U64_arr[i*m+j, :] * V64_arr[j*p+k, :] * W64_arr[k*n+i, :]
                
                for c in range(64):
                    if time.time() > deadline - 2:
                        break
                    keep = [l for l in range(64) if l != c]
                    U_new = U64_arr[:, keep]
                    V_new = V64_arr[:, keep]
                    W_new = W64_arr[:, keep]
                    if verify(U_new, V_new, W_new) < 1e-10:
                        rank = 63
                        best_U, best_V, best_W = U_new, V_new, W_new
                        break
    
    # Final high-precision polish: snap to half-integers if close
    if rank < 66:
        U_pol = best_U.copy()
        V_pol = best_V.copy()
        W_pol = best_W.copy()
        for arr in [U_pol, V_pol, W_pol]:
            arr[np.abs(arr) < 1e-10] = 0.0
            # Snap to nearest half-integer if within tolerance
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if abs(arr[i,j]) > 1e-10:
                        nearest = round(arr[i,j] * 2) / 2
                        if abs(arr[i,j] - nearest) < 1e-8:
                            arr[i,j] = nearest
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
