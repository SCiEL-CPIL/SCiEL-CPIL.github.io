import time
import numpy as np

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 130
    
    # Trivial decomposition (rank = n*m*p = 63)
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
    
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    def _compute_error_fast(U, V, W, n, m, p):
        """Compute total error efficiently using tensor contractions."""
        err = 0.0
        nm = n * m
        mp = m * p
        pn = p * n
        R = U.shape[1]
        
        # Compute all diagonal terms (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        
        # Sample a subset of off-diagonal terms for speed
        # Check terms where indices don't match
        import random
        np.random.seed(42)
        for _ in range(5000):
            a = np.random.randint(nm)
            b = np.random.randint(mp)
            c = np.random.randint(pn)
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if not (i == i2 and j == j2 and k == k2):
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err += val ** 2
        
        return err
    
    def _is_exact(U, V, W, n, m, p, tol=1e-8):
        """Check if the decomposition is exact within tolerance."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Check all diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check all off-diagonal terms
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (i == i2 and j == j2 and k == k2):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        if abs(val) > tol:
                            return False
        
        return True
    
    # Try to find lower-rank decompositions using ALS
    # The key insight: for 3x3 * 3x7, we can try ranks from 62 down
    # Use a smarter initialization and more iterations
    
    for R in range(trivial_rank - 1, 26, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(1000 + R * 10 + trial)
            U = np.random.randn(n * m, R) * 0.3
            V = np.random.randn(m * p, R) * 0.3
            W = np.random.randn(p * n, R) * 0.3
            
            # Alternating least squares with convergence check
            prev_err = float('inf')
            converged = False
            
            for iteration in range(50):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, R))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, R))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass
                
                # Check convergence
                err = _compute_error_fast(U, V, W, n, m, p)
                if err < 1e-10:
                    converged = True
                    break
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
            
            if converged:
                if _is_exact(U, V, W, n, m, p):
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = R
                    found = True
                    break
        
        if found:
            continue  # Try lower rank
        else:
            # If we couldn't find a valid decomposition at this rank,
            # and we've spent enough time, stop
            if time.time() > deadline - 15:
                break
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
