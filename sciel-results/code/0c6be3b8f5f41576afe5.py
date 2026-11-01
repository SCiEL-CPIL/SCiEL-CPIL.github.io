import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 110

    # Target: Rank 11 for 2x3x3
    n2, m2, p2 = 2, 3, 3
    dim_u = n2 * m2
    dim_v = m2 * p2
    dim_w = p2 * n2
    
    # Target tensor T for 2x3x3 is all ones for valid indices
    # But we just check the sum condition.
    
    def find_rank11_233():
        rank = 11
        # Initialize U, V
        # Use Strassen for 2x2x2 part (indices 0,1)
        # Strassen vectors for 2x2x2:
        # u: 7 vectors in R^4 (2x2)
        # v: 7 vectors in R^4
        # w: 7 vectors in R^4
        
        # Standard Strassen decomposition for 2x2 matrix multiplication:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We map these to our U, V, W structure.
        # U indices: i (0,1), j (0,1). 4 entries.
        # V indices: j (0,1), k (0,1). 4 entries.
        # W indices: k (0,1), i (0,1). 4 entries.
        
        # We will initialize U, V with these patterns, and W with zeros, then solve for W.
        # Then iterate.
        
        U = np.zeros((rank, dim_u))
        V = np.zeros((rank, dim_v))
        
        # Fill first 7 vectors with Strassen patterns
        # Note: Our indices are flattened.
        # u_idx = i * m + j. For 2x2, i in {0,1}, j in {0,1}.
        # v_idx = j * p + k. For 2x2, j in {0,1}, k in {0,1}.
        # w_idx = k * n + i. For 2x2, k in {0,1}, i in {0,1}.
        
        def set_u(r, i, j, val):
            U[r, i * m2 + j] = val
            
        def set_v(r, j, k, val):
            V[r, j * p2 + k] = val
            
        # Strassen terms
        # 1: (A00+A11)(B00+B11) -> U: A00,A11; V: B00,B11
        set_u(0, 0, 0, 1); set_u(0, 1, 1, 1)
        set_v(0, 0, 0, 1); set_v(0, 1, 1, 1)
        
        # 2: (A10+A11)B00 -> U: A10,A11; V: B00
        set_u(1, 1, 0, 1); set_u(1, 1, 1, 1)
        set_v(1, 0, 0, 1)
        
        # 3: A00(B01-B11) -> U: A00; V: B01, -B11
        set_u(2, 0, 0, 1)
        set_v(2, 0, 1, 1); set_v(2, 1, 1, -1)
        
        # 4: A11(B10-B00) -> U: A11; V: B10, -B00
        set_u(3, 1, 1, 1)
        set_v(3, 1, 0, 1); set_v(3, 0, 0, -1)
        
        # 5: (A00+A01)B11 -> U: A00,A01; V: B11
        set_u(4, 0, 0, 1); set_u(4, 0, 1, 1)
        set_v(4, 1, 1, 1)
        
        # 6: (A10-A00)(B00+B01) -> U: A10, -A00; V: B00, B01
        set_u(5, 1, 0, 1); set_u(5, 0, 0, -1)
        set_v(5, 0, 0, 1); set_v(5, 0, 1, 1)
        
        # 7: (A01-A11)(B10+B11) -> U: A01, -A11; V: B10, B11
        set_u(6, 0, 1, 1); set_u(6, 1, 1, -1)
        set_v(6, 1, 0, 1); set_v(6, 1, 1, 1)
        
        # Initialize last 4 vectors randomly
        np.random.seed(42)
        for r in range(7, rank):
            U[r, :] = np.random.randn(dim_u)
            V[r, :] = np.random.randn(dim_v)
            
        # Solve for W
        # We need sum_r U_r V_r W_r = T
        # T is 1 for all i,j,k in {0,1,2}
        # This is a linear system for W.
        # Flatten the tensor equation.
        # Number of equations: 2*3*3 = 18.
        # Unknowns: rank * dim_w = 11 * 6 = 66.
        
        A_mat = np.zeros((18, rank * dim_w))
        b_vec = np.ones(18)
        
        eq_idx = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    # Equation: sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i] = 1
                    for r in range(rank):
                        u_val = U[r, i * m2 + j]
                        v_val = V[r, j * p2 + k]
                        if u_val != 0 and v_val != 0:
                            w_idx = r * dim_w + (k * n2 + i)
                            A_mat[eq_idx, w_idx] = u_val * v_val
                    eq_idx += 1
        
        # Solve least squares
        W_flat, residuals, rank_ls, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W = W_flat.reshape(rank, dim_w)
        
        # Check error
        def calc_error(U, V, W):
            err = 0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0
                        for r in range(rank):
                            s += U[r, i * m2 + j] * V[r, j * p2 + k] * W[r, k * n2 + i]
                        err += (1 - s) ** 2
            return np.sqrt(err)
            
        err = calc_error(U, V, W)
        
        # Iterate: Update U, V using gradient descent, then solve for W
        # This might be slow. Let's do a few steps.
        
        for step in range(50):
            if time.time() > deadline:
                break
                
            # Gradient of err^2 w.r.t U, V
            # This is complex. Let's just try random restarts if error is high.
            if err < 1e-6:
                break
            
            # Simple update: add small random noise to U, V and re-solve W
            # This is a heuristic.
            U += np.random.randn(rank, dim_u) * 0.1
            V += np.random.randn(rank, dim_v) * 0.1
            
            A_mat = np.zeros((18, rank * dim_w))
            eq_idx = 0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        for r in range(rank):
                            u_val = U[r, i * m2 + j]
                            v_val = V[r, j * p2 + k]
                            if u_val != 0 and v_val != 0:
                                w_idx = r * dim_w + (k * n2 + i)
                                A_mat[eq_idx, w_idx] = u_val * v_val
                        eq_idx += 1
            
            W_flat, residuals, rank_ls, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W = W_flat.reshape(rank, dim_w)
            err = calc_error(U, V, W)
            
        if err < 1e-6:
            return U, V, W
        return None

    result_233 = find_rank11_233()
    
    if result_233 is not None:
        U233, V233, W233 = result_233
        # Construct 2x3x6 decomposition
        rank_total = 22
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        # Copy first 11 terms
        for r in range(11):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = U233[r, i * m2 + j]
            for j in range(m):
                for k in range(3): # k in 0..2
                    V[j * p + k][r] = V233[r, j * p2 + k]
            for k in range(3): # k in 0..2
                for i in range(n):
                    W[k * n + i][r] = W233[r, k * n2 + i]
                    
        # Copy last 11 terms (shifted k)
        for r in range(11):
            r2 = r + 11
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r2] = U233[r, i * m2 + j]
            for j in range(m):
                for k in range(3, 6): # k in 3..5
                    V[j * p + k][r2] = V233[r, j * p2 + (k - 3)]
            for k in range(3, 6): # k in 3..5
                for i in range(n):
                    W[k * n + i][r2] = W233[r, (k - 3) * n2 + i]
                    
        return {"U": U, "V": V, "W": W}
    else:
        return schoolbook()

