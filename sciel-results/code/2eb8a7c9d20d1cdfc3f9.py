import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Uses block Strassen + direct, then efficient local search to reduce rank.
    """
    n, m, p = 3, 4, 6
    start_time = time.time()
    
    def build_baseline():
        R = 66
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
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
            
            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1
        
        def add_direct_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r
            
            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1
        
        for j in range(3):
            for i in range(2):
                add_direct_1x2_2x2(0, i * 2, i * 2, j * 2, 0, j * 2)
        
        for j in range(3):
            for i in range(2):
                add_strassen_block(1, i * 2, i * 2, j * 2, 1, j * 2)
        
        assert col == R
        return U, V, W
    
    def verify_fast(U_arr, V_arr, W_arr):
        """Fast vectorized verification using numpy."""
        # U_arr: (n*m, r), V_arr: (m*p, r), W_arr: (p*n, r)
        # For each (a,b,c): sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r] == 1
        r = U_arr.shape[1]
        # Build index arrays
        a_idx = np.repeat(np.arange(n), m * p)
        b_idx = np.tile(np.repeat(np.arange(m), p), n)
        c_idx = np.tile(np.arange(p), n * m)
        
        u_idx = a_idx * m + b_idx  # (n*m*p,)
        v_idx = b_idx * p + c_idx  # (n*m*p,)
        w_idx = c_idx * n + a_idx  # (n*m*p,)
        
        # Sum over r: U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
        # This is: sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        products = U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :]
        result = products.sum(axis=1)
        
        return np.allclose(result, np.ones(n * m * p), atol=1e-9)
    
    def try_local_search(U, V, W, deadline):
        """Efficiently try to reduce rank by dropping redundant columns."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            r = U.shape[1]
            
            for drop_idx in range(r):
                if time.time() >= deadline:
                    break
                
                # Remove column drop_idx
                mask = np.ones(r, dtype=bool)
                mask[drop_idx] = False
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]
                
                if verify_fast(U_test, V_test, W_test):
                    U = U_test
                    V = V_test
                    W = W_test
                    improved = True
                    break
        
        # Convert back to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Round to nearest half-integer if close
        def round_half(x):
            r = round(x * 2) / 2
            if abs(x - r) < 1e-9:
                return r
            return x
        
        U_list = [[round_half(v) for v in row] for row in U_list]
        V_list = [[round_half(v) for v in row] for row in V_list]
        W_list = [[round_half(v) for v in row] for row in W_list]
        
        return U_list, V_list, W_list
    
    U, V, W = build_baseline()
    
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    U, V, W = try_local_search(U, V, W, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
