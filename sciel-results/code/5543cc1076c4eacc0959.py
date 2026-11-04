import numpy as np
import time
import scipy.optimize as opt

def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 140

    # Strassen's algorithm for 2x2 matrix multiplication: rank 7
    # For matrices A (2x2), B (2x2), C = A*B (2x2):
    # C[0,0] = M1 - M4 + M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    # where:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])

    # For our 2x3 * 3x6 problem, we partition:
    # A is 2x3, B is 3x6
    # We can split B into two 3x3 blocks: B = [B1 | B2] where B1, B2 are 3x3
    # C = A * B = [A*B1 | A*B2]
    
    # But we need to handle the 2x3 * 3x3 case. Let's think differently.
    # Actually, let's use a more direct approach.
    
    # The 2x3 * 3x6 multiplication can be seen as:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,...,5}
    
    # We can use Strassen-style decomposition by partitioning the middle dimension.
    # Split j into {0} and {1,2}:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # This gives a rank of 2*3*6 = 36 with the trivial decomposition.
    # We need to find a lower rank.
    
    # Let's try to use the fact that we can group terms.
    # For a 2x3 * 3x6 matrix multiplication, one known approach is to use
    # the tensor rank decomposition.
    
    # Let me construct a rank-14 solution using Strassen's idea adapted for this case.
    
    # Actually, let me think about this more carefully. The matrix multiplication
    # tensor for 2x3 * 3x6 has a known rank. Let me try to find it.
    
    # For now, let me try a constructive approach with rank 14.
    
    # The idea: partition the 3x6 matrix B into two 3x3 blocks.
    # B = [B_left | B_right] where B_left and B_right are 3x3.
    # C = A * B = [A*B_left | A*B_right]
    
    # Now, A*B_left is 2x3 * 3x3 = 2x3 result.
    # A*B_right is 2x3 * 3x3 = 2x3 result.
    
    # For each 2x3 * 3x3 multiplication, can we find a low-rank decomposition?
    
    # Let me try a different partitioning. Split the output columns into groups.
    
    # Actually, let me just try to solve this numerically with a better approach.
    
    # Construct the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try rank 14
    def find_decomposition(r, max_time=100):
        t0 = time.time()
        rng = np.random.RandomState(42)
        
        best_U, best_V, best_W = None, None, None
        best_err = np.inf
        
        for restart in range(30):
            if time.time() - t0 > max_time:
                break
            
            U = rng.randn(n*m, r)
            V = rng.randn(m*p, r)
            W = rng.randn(p*n, r)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*r], x[n*m*r:n*m*r+m*p*r], x[n*m*r+m*p*r:]
                U_mat = U_flat.reshape(n*m, r)
                V_mat = V_flat.reshape(m*p, r)
                W_mat = W_flat.reshape(p*n, r)
                
                val = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
                diff = T - val
                return np.sum(diff**2)
            
            res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_err:
                x = res.x
                U_mat = x[:n*m*r].reshape(n*m, r)
                V_mat = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_mat = x[n*m*r+m*p*r:].reshape(p*n, r)
                
                best_err = res.fun
                best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
                
                U_snap = np.round(U_mat * 2) / 2
                V_snap = np.round(V_mat * 2) / 2
                W_snap = np.round(W_mat * 2) / 2
                
                val_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                err_snap = np.sum((T - val_snap)**2)
                
                if err_snap < 1e-9:
                    return U_snap, V_snap, W_snap
        
        if best_err < 1e-9:
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            val_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            err_snap = np.sum((T - val_snap)**2)
            if err_snap < 1e-9:
                return U_snap, V_snap, W_snap
        
        return None
    
    # Try rank 14
    res = find_decomposition(14)
    
    if res is not None:
        U1, V1, W1 = res
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
    else:
        # Fall back to schoolbook rank 36
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
