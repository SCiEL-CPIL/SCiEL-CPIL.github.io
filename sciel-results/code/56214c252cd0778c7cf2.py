import numpy as np
import time

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    time_limit = 115.0

    rank_schoolbook = n * m * p
    best_rank = rank_schoolbook

    def get_schoolbook():
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

    best_solution = get_schoolbook()

    # Precompute the target tensor
    # The target is T[i, j, k, l] = 1 if j == l, else 0
    # This corresponds to matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # Using einsum: T[i,j,k,l] * A[i,j] * B[k,l] = sum_j A[i,j] * B[j,k] when k=l=j
    # Wait, let me re-derive this.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want U, V, W such that:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
    # This is the matrix multiplication tensor.
    # The target tensor T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    
    # Let me think about this differently. The matrix multiplication tensor M is defined by:
    # M[i, j, k] = 1 for all i, j, k (where i in [0,n), j in [0,m), k in [0,p))
    # And we want to decompose M as a sum of rank-1 tensors:
    # M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Wait, that's not the standard decomposition. Let me reconsider.
    
    # Standard tensor decomposition for matrix multiplication:
    # We have A (n x m), B (m x p), and C (n x p)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is in R^{n x m} x R^{m x p} x R^{n x p}
    # T[i,j, k,l] = delta_{j,l}
    # So T[i,j,k,l] = 1 if j == l, else 0
    
    # In the factorization:
    # U has shape (n*m, R) -> U[i*m+j, r]
    # V has shape (m*p, R) -> V[j*p+k, r]  
    # W has shape (p*n, R) -> W[k*n+i, r]
    
    # The reconstruction is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,k}
    
    # Wait, that doesn't look right either. Let me look at the original code more carefully.
    
    # From the original code:
    # U[i * m + j][col] = 1  -> U[i*m+j, r]
    # V[j * p + k][col] = 1  -> V[j*p+k, r]
    # W[k * n + i][col] = 1  -> W[k*n+i, r]
    
    # And the einsum was: 'ijr,jkr,kir->ijjk'
    # U: i, j, r (shape n x m x R)
    # V: j, k, r (shape m x p x R)
    # W: k, i, r (shape p x n x R)
    # Output: i, j, j, k (shape n x m x m x p)
    
    # So the reconstruction is:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = target[i,j,j,k]
    
    # And target[i,j,j,k] = 1 for all i,j,k
    
    # So the target tensor T[i,j,k,l] = 1 if j == l, else 0? 
    # No, target[i,j,j,k] = 1 means the 2nd and 3rd indices are both j.
    # So T[i, a, b, c] = 1 if a == b and c is free? 
    # Actually, T[i, j, j, k] = 1 means T[i, a, b, c] = 1 if a == b, regardless of c.
    # Wait no, the 4th index is k, which varies independently.
    # So T[i, a, b, c] = 1 if a == b, for any c.
    
    # Let me verify: for fixed i, the slice T[i, :, :, :] is a 3D tensor where
    # T[i, a, b, c] = 1 if a == b, else 0.
    # This is like a "diagonal" in the first two indices of the 3D slice.
    
    # OK so the target is: T[i, j, k, l] = 1 if j == k, else 0
    # (where the indices are i in [0,n), j in [0,m), k in [0,m), l in [0,p))
    
    # Let me precompute this target
    target = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, j, k] = 1.0

    # Try to find lower rank decompositions
    # Strategy: random U, V, then solve linear system for W
    
    for R in range(rank_schoolbook - 1, 50, -1):
        if time.time() - start_time > time_limit - 5:
            break
        
        found = False
        num_attempts = 20 if R > 80 else 50
        
        for attempt in range(num_attempts):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Random U, V with small integers
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            # Build linear system for W
            # The equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = target[i,j,j,k]
            # For each (i, j, k), this is a linear equation in W.
            # Number of equations: n * m * p
            # Number of unknowns: p * n * R
            
            num_eqs = n * m * p
            num_vars = p * n * R
            
            A = np.zeros((num_eqs, num_vars))
            b = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        b[eq_idx] = 1.0  # target[i, j, j, k] = 1
                        
                        for r in range(R):
                            var_idx = (k * n + i) * R + r
                            A[eq_idx, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                        
                        eq_idx += 1
            
            # Solve the linear system
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if W is close to integers or half-integers
            W_int = np.round(W_flat).astype(int)
            W_half = np.round(W_flat * 2).astype(int)
            
            # Verify integer solution
            U_check = U.reshape(n, m, R)
            V_check = V.reshape(m, p, R)
            W_check = W_int.reshape(p, n, R)
            
            T = np.einsum('ijr,jkr,kir->ijjk', U_check, V_check, W_check)
            
            if np.allclose(T, target, atol=1e-8):
                best_rank = R
                best_solution = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_int.reshape(p * n, R).tolist()
                }
                found = True
                break
            
            # Verify half-integer solution
            W_check = (W_half / 2.0).reshape(p, n, R)
            T = np.einsum('ijr,jkr,kir->ijjk', U_check, V_check, W_check)
            
            if np.allclose(T, target, atol=1e-8):
                best_rank = R
                best_solution = {
                    "U": (U * 2).tolist(),
                    "V": (V * 2).tolist(),
                    "W": W_half.reshape(p * n, R).tolist()
                }
                found = True
                break
        
        if found:
            pass

    return best_solution
