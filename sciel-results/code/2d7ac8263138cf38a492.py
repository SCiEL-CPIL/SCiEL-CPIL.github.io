import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p tensor decomposition for M_{n,m,p}."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that U, V, W form a valid tensor decomposition of M_{n,m,p}.
    """
    r = len(U[0]) if U else 0
    if r != len(V[0]) or r != len(W[0]):
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)

    # Check all (i,j,k) on-diagonal terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # sum_r U[a,r] V[b,r] W[c,r]
                val = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                if abs(val - 1.0) > tol:
                    return False

    # Check a subset of off-diagonal terms to ensure they are 0
    valid_combos = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_combos.add((i*m+j, j*p+k, k*n+i))

    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) in valid_combos:
                    continue
                val = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                if abs(val) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # The rank of 3x3x7 is suspected to be 21.
    # We search from 21 upwards to 40.
    # If we find a lower rank, that's even better, but 21 is a strong candidate.
    
    for R in range(21, 41):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        for trial in range(20):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + trial + R)
            
            # Initialize with small random values
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # ALS iterations
            converged = False
            prev_err = float('inf')
            
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        M = np.zeros((p, R))
                        for k in range(p):
                            b = j * p + k
                            c = k * n + i
                            M[k, :] = V[b, :] * W[c, :]
                        try:
                            # Add regularization to avoid singularity
                            reg = 1e-6 * np.eye(R)
                            u, _, _, _ = np.linalg.lstsq(M + reg, np.ones(p), rcond=None)
                            U[a, :] = u
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        b = j * p + k
                        M = np.zeros((n, R))
                        for i in range(n):
                            a = i * m + j
                            c = k * n + i
                            M[i, :] = U[a, :] * W[c, :]
                        try:
                            reg = 1e-6 * np.eye(R)
                            v, _, _, _ = np.linalg.lstsq(M + reg, np.ones(n), rcond=None)
                            V[b, :] = v
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        c = k * n + i
                        M = np.zeros((m, R))
                        for j in range(m):
                            a = i * m + j
                            b = j * p + k
                            M[j, :] = U[a, :] * V[b, :]
                        try:
                            reg = 1e-6 * np.eye(R)
                            w, _, _, _ = np.linalg.lstsq(M + reg, np.ones(m), rcond=None)
                            W[c, :] = w
                        except:
                            pass
                
                # Check convergence / error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        for k in range(p):
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a], V[b] * W[c])
                            err += (val - 1.0) ** 2
                
                # Also check off-diagonal errors roughly to avoid converging to wrong subspace
                # This is expensive, so we only do it every few iterations or if err is low
                if iteration % 10 == 0 or err < 1e-3:
                    off_err = 0.0
                    valid_combos = set()
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                valid_combos.add((i2*m+j2, j2*p+k2, k2*n+i2))
                    
                    # Sample some off-diagonals
                    for _ in range(100):
                        a = np.random.randint(n*m)
                        b = np.random.randint(m*p)
                        c = np.random.randint(p*n)
                        if (a,b,c) in valid_combos: continue
                        val = np.dot(U[a], V[b] * W[c])
                        off_err += val*val
                    
                    if off_err > 1.0: # If off-diagonal errors are large, don't declare convergence
                        continue 
                
                if abs(err - prev_err) < 1e-10:
                    converged = True
                    break
                prev_err = err
                
                if err < 1e-8:
                    converged = True
                    break
            
            if converged or err < 1e-6:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                if _verify(n, m, p, U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    
    return best_result
# EVOLVE-BLOCK-END
