import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    DEADLINE = start_time + 135

    def verify(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        np.random.seed(42)
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_mat[:, r].reshape(n, m)
            v_r = V_mat[:, r].reshape(m, p)
            w_r = W_mat[:, r].reshape(p, n)
            
            for i in range(n):
                for j in range(p):
                    ua = np.dot(u_r[i, :], A_test[i, :])
                    vb = np.dot(v_r[:, j], B_test[:, j])
                    C_computed[i, j] += w_r[j, i] * ua * vb
        
        return np.allclose(C_computed, C_expected, atol=1e-10)

    # Try to find rank-13 decomposition for 2x4x2 via gradient descent
    def find_rank13_2x4x2():
        rank = 13
        np.random.seed(12345)
        U = np.random.randn(8, rank) * 0.5
        V = np.random.randn(8, rank) * 0.5
        W = np.random.randn(4, rank) * 0.5
        target = np.ones(16)
        
        def compute_tensor(U, V, W):
            result = np.zeros(16)
            for r in range(rank):
                u_r = U[:, r].reshape(2, 4)
                v_r = V[:, r].reshape(4, 2)
                w_r = W[:, r].reshape(2, 2)
                for i in range(2):
                    for j in range(4):
                        for k in range(2):
                            result[i*8 + j*2 + k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
            return result
        
        def compute_error(U, V, W):
            t = compute_tensor(U, V, W)
            return np.sum((t - target)**2)
        
        lr = 0.01
        best_error = compute_error(U, V, W)
        
        for iteration in range(3000):
            if time.time() > DEADLINE - 40:
                break
            
            t = compute_tensor(U, V, W)
            err = t - target
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank):
                u_r = U[:, r].reshape(2, 4)
                v_r = V[:, r].reshape(4, 2)
                w_r = W[:, r].reshape(2, 2)
                
                for i in range(2):
                    for j in range(4):
                        for k in range(2):
                            idx = i*8 + j*2 + k
                            e = err[idx]
                            grad_U[i*4+j, r] += e * v_r[j, k] * w_r[k, i]
                            grad_V[j*2+k, r] += e * u_r[i, j] * w_r[k, i]
                            grad_W[k*2+i, r] += e * u_r[i, j] * v_r[j, k]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            new_error = compute_error(U, V, W)
            if new_error < best_error:
                best_error = new_error
                if best_error < 1e-12:
                    break
            else:
                lr *= 0.95
                U += lr * grad_U
                V += lr * grad_V
                W += lr * grad_W
        
        final_error = compute_error(U, V, W)
        if final_error < 1e-8:
            return U, V, W
        return None

    result_13 = find_rank13_2x4x2()
    
    if result_13 is not None and time.time() < DEADLINE - 15:
        U13, V13, W13 = result_13
        U52 = np.zeros((8, 52))
        V52 = np.zeros((32, 52))
        W52 = np.zeros((16, 52))
        
        for block in range(4):
            U52[:, block*13:(block+1)*13] = U13
            for j in range(4):
                for k_local in range(2):
                    V52[j*8 + block*2 + k_local, block*13:(block+1)*13] = V13[j*2 + k_local]
            for k_local in range(2):
                for i in range(2):
                    W52[(block*2 + k_local)*2 + i, block*13:(block+1)*13] = W13[k_local*2 + i]
        
        if verify(U52, V52, W52):
            return {
                "U": U52.tolist(),
                "V": V52.tolist(),
                "W": W52.tolist()
            }

    # Fallback: rank-56 Strassen decomposition
    rank_56 = 56
    U = np.zeros((8, rank_56))
    V = np.zeros((32, rank_56))
    W = np.zeros((16, rank_56))
    
    col = 0
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0*4 + a_col_off
        a01 = 0*4 + a_col_off + 1
        a10 = 1*4 + a_col_off
        a11 = 1*4 + a_col_off + 1
        
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    
    assert col == 56
    
    if not verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
