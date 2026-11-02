import numpy as np
import time
import scipy.optimize as opt

def construct():
    n, m, p = 3, 3, 6
    deadline = time.time() + 110  # Safety margin
    
    def verify_3x3(U, V, W, R):
        # U, V, W are (9, R)
        # Reshape to (3, 3, R)
        U_3 = U.reshape(3, 3, R)
        V_3 = V.reshape(3, 3, R)
        W_3 = W.reshape(3, 3, R)
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[i,k,r]
        # S[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[i,k,r]
        # Note: V index is j,k. In V_3, row is j, col is k?
        # In the loop: V[k*3+j] was used. 
        # Wait, the original code used V[k*3+j]. 
        # In standard matrix mult C = AB, C[i,k] = sum_j A[i,j] B[j,k].
        # So V should be indexed by [j,k].
        # Original code: V[k*3+j]. This implies V is stored column-major or transposed?
        # Let's check the original verify:
        # s += U[i*3+k, r] * V[k*3+j, r] * W[i*3+j, r]
        # This corresponds to C[i,j] = sum_k A[i,k] B[k,j].
        # So A is U, B is V, C is W.
        # A[i,k] -> U[i*3+k]
        # B[k,j] -> V[k*3+j]
        # C[i,j] -> W[i*3+j]
        # This is standard row-major storage for A and C, and row-major for B?
        # B[k,j] is row k, col j.
        # So U is (3,3), V is (3,3), W is (3,3).
        # U_3[i,k,r], V_3[k,j,r], W_3[i,j,r].
        
        # Using einsum:
        # S[i,j] = sum_k sum_r U[i,k,r] V[k,j,r] W[i,j,r]
        S = np.einsum('ikr,kjr,ijr->ij', U_3, V_3, W_3)
        
        # Check if S is all 1s
        if not np.allclose(S, np.ones((3,3)), atol=1e-5):
            return False
        return True

    def residuals(params, R):
        # params: flat array of size 27*R
        U = params[0:9*R].reshape(3, 3, R)
        V = params[9*R:18*R].reshape(3, 3, R)
        W = params[18*R:27*R].reshape(3, 3, R)
        
        # S[i,j] = sum_k sum_r U[i,k,r] V[k,j,r] W[i,j,r]
        S = np.einsum('ikr,kjr,ijr->ij', U, V, W)
        
        # Residuals are S - 1
        return (S - 1.0).flatten()

    def try_find_decomp(R, max_restarts=50):
        best_sol = None
        best_err = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Sparse Random Initialization
            # Each column of U, V, W has ~3 non-zeros
            params = np.zeros(27 * R)
            
            # Initialize U
            for r in range(R):
                # Pick 3 random indices for U[:, r]
                idx = np.random.choice(9, 3, replace=False)
                params[idx + r*0] = np.random.choice([-1, 1], 3) # This is wrong indexing for flat array
                # Correct indexing:
                # U is 9*R. U[i, r] is at i + 9*r? No, reshape(9, R) means row-major.
                # U[i, r] is at i * R + r? No.
                # np.reshape(9, R) from flat array:
                # element (i, r) is at i * R + r.
                # So for column r, indices are i * R + r for i in 0..8.
                pass
            
            # Let's do it properly
            U_flat = np.zeros(9 * R)
            V_flat = np.zeros(9 * R)
            W_flat = np.zeros(9 * R)
            
            for r in range(R):
                idx_u = np.random.choice(9, 3, replace=False)
                idx_v = np.random.choice(9, 3, replace=False)
                idx_w = np.random.choice(9, 3, replace=False)
                
                U_flat[idx_u * R + r] = np.random.choice([-1, 1], 3)
                V_flat[idx_v * R + r] = np.random.choice([-1, 1], 3)
                W_flat[idx_w * R + r] = np.random.choice([-1, 1], 3)
            
            params = np.concatenate([U_flat, V_flat, W_flat])
            
            # Optimization
            try:
                sol = opt.minimize(
                    lambda x: np.sum(residuals(x, R)**2), 
                    params, 
                    method='L-BFGS-B', 
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False}
                )
                if sol.fun < best_err:
                    best_err = sol.fun
                    best_sol = sol.x
                    
                    if best_err < 1e-8:
                        U = best_sol[0:9*R].reshape(9, R)
                        V = best_sol[9*R:18*R].reshape(9, R)
                        W = best_sol[18*R:27*R].reshape(9, R)
                        
                        # Round to nearest half-integer
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        
                        if verify_3x3(U_r, V_r, W_r, R):
                            return U_r, V_r, W_r
            except Exception:
                continue
        return None

    # Try ranks 23 to 26
    for R in range(23, 27):
        if time.time() > deadline - 20:
            break
        result = try_find_decomp(R)
        if result:
            U3, V3, W3 = result
            
            # Construct 3x3x6 decomposition
            U_total = np.zeros((9, 2*R))
            V_total = np.zeros((18, 2*R))
            W_total = np.zeros((18, 2*R))
            
            # First block
            U_total[:, :R] = U3
            V_total[:9, :R] = V3
            W_total[:9, :R] = W3
            
            # Second block
            U_total[:, R:] = U3
            V_total[9:, R:] = V3
            W_total[9:, R:] = W3
            
            # Remap W for 3x3x6 convention
            # W_total index: k*n + i = k*3 + i
            # W3 index: i*3 + j (where j is col of C1)
            # For C1 (k=0,1,2): W3[i*3+j] -> W_total[j*3+i]
            # For C2 (k=3,4,5): W3[i*3+j] -> W_total[(j+3)*3+i]
            
            W_total_mapped = np.zeros((18, 2*R))
            for i in range(3):
                for j in range(3):
                    val = W3[i*3+j, :]
                    W_total_mapped[j*3+i, :R] = val
                    W_total_mapped[(j+3)*3+i, R:] = val
                    
            W_total = W_total_mapped
            
            # Verify full
            def verify_full(U, V, W, n, m, p):
                # U: (9, Rank), V: (18, Rank), W: (18, Rank)
                # U[i*m+j], V[j*p+k], W[k*n+i]
                U_3 = U.reshape(3, 3, U.shape[1])
                V_3 = V.reshape(3, 6, V.shape[1])
                W_3 = W.reshape(6, 3, W.shape[1])
                
                # S[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Note: W index is k*n + i. So W_3[k, i, r].
                S = np.einsum('ijr,jkr,kr->ik', U_3, V_3, W_3)
                
                if not np.allclose(S, np.ones((3,6)), atol=1e-5):
                    return False
                return True

            if verify_full(U_total, V_total, W_total, n, m, p):
                return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
