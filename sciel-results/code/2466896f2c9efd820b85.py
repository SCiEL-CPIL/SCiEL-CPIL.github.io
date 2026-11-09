import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build the linear system for W
    # For each (i,j,k): sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
    # This is a linear system: A @ w = b where w is the vectorized W
    
    # Try different ranks
    for rank in range(62, 20, -1):
        if time.time() > deadline:
            break
        for trial in range(20):
            if time.time() > deadline:
                break
            np.random.seed(trial * 100000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            
            # Build the linear system for W
            # Number of equations: n*m*p = 63
            # Number of unknowns: p*n*rank = 21*rank
            num_eqs = n * m * p
            num_unk = p * n * rank
            
            if num_unk < num_eqs:
                continue  # Not enough degrees of freedom
            
            A = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            A[eq_idx, (k*n+i)*rank + r] = U[i*m+j, r] * V[j*p+k, r]
                        eq_idx += 1
            
            # Solve the linear system
            try:
                w, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if the solution is exact
                if len(residuals) > 0 and residuals[0] < 1e-10:
                    # Reshape w to W
                    W = w.reshape((p*n, rank))
                    # Verify
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                if abs(val - 1) > 1e-10:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            except:
                pass
    
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
