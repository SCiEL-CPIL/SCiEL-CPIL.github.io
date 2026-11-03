import numpy as np

def construct():
    n, m, p = 3, 3, 7
    # Try rank 21
    R = 21
    # A is R x (n*m) = 21 x 9
    # B is R x (m*p) = 21 x 21
    # W is R x (p*n) = 21 x 21
    
    for seed in range(100):
        np.random.seed(seed)
        A = np.random.randint(-1, 2, size=(R, n*m))
        B = np.random.randint(-1, 2, size=(R, m*p))
        
        # Check if we can solve for W
        W = np.zeros((R, p*n))
        success = True
        for i in range(n):
            for j in range(p):
                # Construct M of size (m*m) x R = 9 x 21
                M = np.zeros((m*m, R))
                for k in range(m):
                    for kp in range(m):
                        idx = k * m + kp
                        # M[idx, r] = A[r, i*m + k] * B[r, kp*p + j]
                        M[idx, :] = A[:, i*m + k] * B[:, kp*p + j]
                
                # We need M w = vec(I_m)
                b = np.eye(m).flatten()
                
                # Check rank
                rank = np.linalg.matrix_rank(M)
                if rank < m*m:
                    success = False
                    break
                
                # Solve least squares
                w, residuals, rank2, s = np.linalg.lstsq(M, b, rcond=None)
                
                # Check if solution is exact
                if not np.allclose(M @ w, b):
                    success = False
                    break
                
                # Check if w is close to half-integers
                w_check = w * 2
                if not np.allclose(w_check, np.round(w_check)):
                    success = False
                    break
                
                W[:, j*n + i] = w # Note: W index is k*n+i? No, W[k*n+i, r] in prompt?
                # Prompt says W[k*n+i, r]. Here i is row of C, j is col of C.
                # C is p x n. So index is j*n + i.
        
        if success:
            return {"U": A.tolist(), "V": B.tolist(), "W": W.tolist()}
            
    # Fallback to rank 63
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
