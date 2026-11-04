import time
import numpy as np

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140
    
    # Trivial decomposition
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Function to get 3x3x3 tensor
    def get_333_tensor():
        T = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    T[a, b, c] = 1
        return T

    # Function to find rank 21 decomposition for 3x3x3
    def find_333_decomposition():
        T = get_333_tensor()
        R = 21
        # Random init
        for seed in range(10):
            np.random.seed(seed)
            U = np.random.randn(9, R) * 0.1
            V = np.random.randn(9, R) * 0.1
            W = np.random.randn(9, R) * 0.1
            
            for iter in range(100):
                # Update U
                for a in range(9):
                    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                    # For fixed a, we want sum_r U[a, r] (V[b, r] W[c, r]) = T[a, b, c]
                    # This is a linear system for U[a, :]
                    # Let X[b, c] = T[a, b, c]
                    # Let Y[b, c] = V[b, :] * W[c, :] (element-wise? No, it's a bilinear form)
                    # We want U[a, :] such that sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
                    # This can be viewed as:
                    # vec(U[a, :]^T V W^T) ? No.
                    # Let's form the matrix A of size (81, R)
                    # A[ (b, c), r ] = V[b, r] * W[c, r]
                    # Then A @ U[a, :]^T = T[a, :, :].flatten()
                    A = np.zeros((81, R))
                    for b in range(9):
                        for c in range(9):
                            A[b * 9 + c, :] = V[b, :] * W[c, :]
                    try:
                        U[a, :], _, _, _ = np.linalg.lstsq(A, T[a, :, :].flatten(), rcond=None)
                    except:
                        pass
                
                # Update V
                for b in range(9):
                    A = np.zeros((81, R))
                    for a in range(9):
                        for c in range(9):
                            A[a * 9 + c, :] = U[a, :] * W[c, :]
                    try:
                        V[b, :], _, _, _ = np.linalg.lstsq(A, T[:, b, :].flatten(), rcond=None)
                    except:
                        pass
                
                # Update W
                for c in range(9):
                    A = np.zeros((81, R))
                    for a in range(9):
                        for b in range(9):
                            A[a * 9 + b, :] = U[a, :] * V[b, :]
                    try:
                        W[c, :], _, _, _ = np.linalg.lstsq(A, T[:, :, c].flatten(), rcond=None)
                    except:
                        pass
                
                # Check convergence
                err = 0
                for r in range(R):
                    err += np.sum(np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]) - T)**2
                if err < 1e-10:
                    break
            
            # Round and verify
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_recon = np.zeros((9, 9, 9))
            for r in range(R):
                T_recon += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
            
            if np.allclose(T_recon, T, atol=1e-10):
                return U_r, V_r, W_r
        
        return None

    U333, V333, W333 = find_333_decomposition()
    
    if U333 is not None:
        # Construct 3x3x7 decomposition
        R_total = 21 + 21 + 8
        U = np.zeros((9, R_total))
        V = np.zeros((21, R_total))
        W = np.zeros((21, R_total))
        
        # Block 1: 3x3x3 for cols 0-2
        U[:, 0:21] = U333
        for j in range(3):
            for k in range(3):
                b_global = j * 7 + k
                b_local = j * 3 + k
                V[b_global, 0:21] = V333[b_local, :]
        for k in range(3):
            for i in range(3):
                c_global = k * 3 + i
                c_local = k * 3 + i
                W[c_global, 0:21] = W333[c_local, :]
        
        # Block 2: 3x3x3 for cols 3-5
        U[:, 21:42] = U333
        for j in range(3):
            for k in range(3):
                b_global = j * 7 + (k + 3)
                b_local = j * 3 + k
                V[b_global, 21:42] = V333[b_local, :]
        for k in range(3):
            for i in range(3):
                c_global = (k + 3) * 3 + i
                c_local = k * 3 + i
                W[c_global, 21:42] = W333[c_local, :]
        
        # Block 3: 3x3x1 for col 6
        # Rank 8
        for a in range(9):
            r = 42 + a
            U[a, r] = 1
            j = a % 3
            i = a // 3
            b_global = j * 7 + 6
            V[b_global, r] = 1
            c_global = 6 * 3 + i
            W[c_global, r] = 1
            
        # Verify
        T_recon = np.zeros((9, 21, 21))
        for r in range(R_total):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        T_target = np.zeros((9, 21, 21))
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    a = i * 3 + j
                    b = j * 7 + k
                    c = k * 3 + i
                    T_target[a, b, c] = 1
        
        if np.allclose(T_recon, T_target, atol=1e-10):
            return {
                "U": to_list(U),
                "V": to_list(V),
                "W": to_list(W)
            }
            
    return {
        "U": to_list(U_trivial),
        "V": to_list(V_trivial),
        "W": to_list(W_trivial)
    }
