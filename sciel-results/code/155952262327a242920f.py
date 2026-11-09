import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 21, 14
    
    # Schoolbook fallback
    rank_school = n*m*p  # 42
    U_s = np.zeros((nm, rank_school))
    V_s = np.zeros((mp, rank_school))
    W_s = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    def verify(U, V, W, r):
        """Check if decomposition is exact using einsum."""
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # C[i, k, j, jp] = sum_c U[i, j, c] * V[jp, k, c] * W[k, i, c]
        C = np.einsum('ijc,jkc,kic->ikjj', U_reshaped, V_reshaped, W_reshaped)
        
        I = np.eye(m)
        C_flat = C.reshape(n, p, m*m)
        I_flat = I.reshape(m*m)
        
        return np.allclose(C_flat, I_flat, atol=1e-8)
    
    def try_rank(r, max_trials=500):
        # Adjust trials based on rank likelihood
        if r < 30:
            max_trials = 100
        else:
            max_trials = 2000
            
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(42 + r * 10000 + trial)
            # Use integer/half-integer values
            U = 0.5 * np.random.randint(-6, 7, size=(nm, r))
            V = 0.5 * np.random.randint(-6, 7, size=(mp, r))
            
            # Check if for each (i,k), the 9×r matrix has rank 9
            solvable = True
            W = np.zeros((pn, r))
            
            for i in range(n):
                for k in range(p):
                    # Build 9×r matrix
                    B = np.zeros((m*m, r))
                    for j in range(m):
                        for jp in range(m):
                            idx = j*m + jp
                            B[idx, :] = U[i*m+j, :] * V[jp*p+k, :]
                    
                    # Target: vec(I_m)
                    target = np.zeros(m*m)
                    for j in range(m):
                        target[j*m+j] = 1.0
                    
                    # Solve B @ w = target
                    if np.linalg.matrix_rank(B) < m*m:
                        solvable = False
                        break
                    
                    # Least squares solution
                    w, res, rank, sv = np.linalg.lstsq(B, target, rcond=1e-10)
                    W[k*n+i, :] = w
                    # Verify
                    if np.linalg.norm(B @ w - target) > 1e-8:
                        solvable = False
                        break
                
                if not solvable:
                    break
            
            if solvable:
                # Final verification
                if verify(U, V, W, r):
                    return U, V, W
        
        return None
    
    # Try ranks from lower bound up
    for r in range(max(nm, mp, pn), rank_school):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            break
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
