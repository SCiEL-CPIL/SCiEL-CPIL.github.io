import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if we are computing C[i, k] from A[i, j] and B[j, k]
    # The multiplication tensor is T_{i, j, k} = A_{i, j} B_{j, k}
    # In the decomposition, we want U, V, W such that:
    # T = sum_r U_r \otimes V_r \otimes W_r
    # Where U_r is a vector of length n*m? No.
    # The standard form:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}
    # Decomposition:
    # C = sum_{r=1}^R (sum_{i,j} U_{i,j}^r A_{i,j}) (sum_{j,k} V_{j,k}^r B_{j,k}) * W_{i,k}^r ?
    # No, the factors are:
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # The reconstruction is:
    # Res_{i, k} = sum_{r} W_{k*n+i, r} * (sum_{j} U_{i*m+j, r} A_{i,j}) * (sum_{j} V_{j*p+k, r} B_{j,k})
    # Wait, the indices in the prompt are:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n+i, r]
    # So the term is W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    # Summing over r, i, j, k?
    # The output is C[i, k].
    # C[i, k] = sum_{r=1}^R W[k*n+i, r] * (sum_{j=1}^m U[i*m+j, r] A_{i,j}) * (sum_{j=1}^m V[j*p+k, r] B[j,k])
    # This must equal sum_{j=1}^m A[i,j] B[j,k]
    
    # Let's define the target tensor T_{i, j, k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # Actually, the tensor of multiplication is $M_{n,m,p}$ with entries $M_{i,j,k} = 1$ for all $i,j,k$?
    # No, the tensor $T$ such that $C = T(A, B)$.
    # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # This corresponds to the tensor $T$ with indices $i, j, k$ where $T_{i,j,k} = 1$.
    # Wait, the decomposition is:
    # $T_{i,j,k} = \sum_{r} U_{i,j}^r V_{j,k}^r W_{i,k}^r$?
    # The prompt says:
    # U: (n*m, rank) -> index i*m+j
    # V: (m*p, rank) -> index j*p+k
    # W: (p*n, rank) -> index k*n+i
    # So the equation is:
    # $1 = \sum_{r=1}^R U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$ for all $i,j,k$.
    # This is the condition for the decomposition to be exact.

    target = np.ones((n, m, p))
    
    best_result = None
    start_time = time.time()
    
    # Try ranks from lower bound to schoolbook
    lower_bound = max(n*m, m*p, p*n)
    schoolbook = n*m*p
    
    for R in range(lower_bound, schoolbook + 1):
        if time.time() - start_time > 140:
            break
            
        success = False
        for restart in range(5):
            if time.time() - start_time > 145:
                break
                
            # Initialize U, V, W
            # Use small random integers or floats
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
                U_mat = U_flat.reshape(n*m, R)
                V_mat = V_flat.reshape(m*p, R)
                W_mat = W_flat.reshape(p*n, R)
                
                # Compute reconstruction
                # Res_{i, j, k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # We can use einsum
                # U: (nm, R), V: (mp, R), W: (pn, R)
                # We need T_{i,j,k}
                # Let's build the tensor T_rec
                # T_rec = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat) ? No.
                # Indices:
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                # This coupling of indices (j in U and V, k in V and W, i in W and U) makes simple einsum tricky without reshaping.
                
                # Reshape U to (n, m, R)
                U_3d = U_mat.reshape(n, m, R)
                # Reshape V to (m, p, R)
                V_3d = V_mat.reshape(m, p, R)
                # Reshape W to (p, n, R)
                W_3d = W_mat.reshape(p, n, R)
                
                # T_rec[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                # Note: V is indexed by j, k. W by k, i. U by i, j.
                T_rec = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                
                # Error
                err = np.sum((T_rec - target)**2)
                
                # Penalty for large values to encourage small integers
                penalty = 0.01 * (np.sum(U_mat**2) + np.sum(V_mat**2) + np.sum(W_mat**2))
                
                return err + penalty
            
            # Optimize
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200})
            
            if res.success and res.fun < 1e-4:
                # Extract and round
                U_flat, V_flat, W_flat = res.x[:n*m*R], res.x[n*m*R:n*m*R+m*p*R], res.x[n*m*R+m*p*R:]
                U_mat = (U_flat.reshape(n*m, R) * 2).round() / 2.0
                V_mat = (V_flat.reshape(m*p, R) * 2).round() / 2.0
                W_mat = (W_flat.reshape(p*n, R) * 2).round() / 2.0
                
                # Verify exactness
                U_3d = U_mat.reshape(n, m, R)
                V_3d = V_mat.reshape(m, p, R)
                W_3d = W_mat.reshape(p, n, R)
                T_rec = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                
                if np.allclose(T_rec, target, atol=1e-9):
                    # Found a valid decomposition
                    return {
                        "U": U_mat.tolist(),
                        "V": V_mat.tolist(),
                        "W": W_mat.tolist()
                    }
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
