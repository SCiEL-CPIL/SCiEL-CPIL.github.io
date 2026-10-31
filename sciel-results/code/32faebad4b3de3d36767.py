import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = 4, 4, 4

    # 2x2 Strassen factors
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00 + A11) * (B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10 + A11) * B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M2 = A00 * (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M3 = A11 * (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M4 = (A00 + A01) * B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1

    # Build 49-rank decomposition
    rank = 49
    U_arr = np.zeros((16, 49), dtype=int)
    V_arr = np.zeros((16, 49), dtype=int)
    W_arr = np.zeros((16, 49), dtype=int)
    
    I_map = [i // 2 for i in range(4)]
    J_map = [j // 2 for j in range(4)]
    ii_map = [i % 2 for i in range(4)]
    jj_map = [j % 2 for j in range(4)]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for i in range(4):
                for j in range(4):
                    coeff = U2[I_map[i]*2 + J_map[j], s] * U2[ii_map[i]*2 + jj_map[j], t]
                    if coeff != 0:
                        U_arr[i*4 + j, r] = coeff
            for k in range(4):
                for l in range(4):
                    coeff = V2[I_map[k]*2 + J_map[l], s] * V2[ii_map[k]*2 + jj_map[l], t]
                    if coeff != 0:
                        V_arr[k*4 + l, r] = coeff
            for j in range(4):
                for i in range(4):
                    coeff = W2[I_map[i]*2 + J_map[j], s] * W2[ii_map[i]*2 + jj_map[j], t]
                    if coeff != 0:
                        W_arr[j*4 + i, r] = coeff
    
    # Fast vectorized verification
    def verify_decomp_fast(U_mat, V_mat, W_mat, n, m, p):
        """Verify decomposition using vectorized operations."""
        U_mat = np.array(U_mat, dtype=float)
        V_mat = np.array(V_mat, dtype=float)
        W_mat = np.array(W_mat, dtype=float)
        rank = U_mat.shape[1]
        
        np.random.seed(42)
        A = np.random.randint(-5, 5, (n, m)).astype(float)
        B = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A @ B
        
        # Reshape for vectorized computation
        U_block = U_mat.reshape(n, m, rank)  # U[i, a, r]
        V_block = V_mat.reshape(m, p, rank)  # V[a, j, r]
        W_block = W_mat.reshape(p, n, rank)  # W[j, i, r]
        
        # For each r: C[i,j] += sum_a U[i,a,r] * V[a,j,r] * W[j,i,r]
        # Vectorize: compute all r at once
        # intermediate[i, a, j, r] = U[i,a,r] * V[a,j,r]
        # C[i,j] = sum_{a,r} intermediate[i,a,j,r] * W[j,i,r]
        
        # More efficient: for each r, compute outer product
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_mat[:, r].reshape(n, m)
            v_r = V_mat[:, r].reshape(m, p)
            w_r = W_mat[:, r].reshape(p, n)
            # C[i,j] += sum_a u_r[i,a] * v_r[a,j] * w_r[j,i]
            # = sum_a (u_r[i,a] * v_r[a,j]) * w_r[j,i]
            # Let M_r = u_r @ v_r, then C[i,j] += sum_a M_r[i,a]... no
            # Actually: sum_a u_r[i,a] * v_r[a,j] is (u_r @ v_r)[i,j]
            # But we need sum_a u_r[i,a] * v_r[a,j] * w_r[j,i]
            # This is not simply (u_r @ v_r)[i,j] * w_r[j,i]
            # It's: for each (i,j), sum over a of u_r[i,a]*v_r[a,j]*w_r[j,i]
            # = w_r[j,i] * sum_a u_r[i,a]*v_r[a,j] = w_r[j,i] * (u_r @ v_r)[i,j]
            M_r = u_r @ v_r  # (n, p)
            C_computed += M_r * w_r.T  # w_r is (p,n), w_r.T is (n,p), so w_r.T[i,j] = w_r[j,i]
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if verify_decomp_fast(U_arr, V_arr, W_arr, n, m, p):
        # Try to find rank 48 by removing columns
        for remove_col in range(49):
            if time.time() > deadline:
                break
            
            U_test = np.delete(U_arr, remove_col, axis=1)
            V_test = np.delete(V_arr, remove_col, axis=1)
            W_test = np.delete(W_arr, remove_col, axis=1)
            
            if verify_decomp_fast(U_test, V_test, W_test, n, m, p):
                rank = 48
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                break
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
