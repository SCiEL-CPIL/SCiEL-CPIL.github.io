import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p):
    """Check if the factorization is exact."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _to_rational(U, V, W, tol=1e-9):
    """Convert float matrices to rational (half-integer) entries if possible."""
    from fractions import Fraction
    
    def convert_matrix(M):
        result = []
        for row in M:
            new_row = []
            for val in row:
                if abs(val) < tol:
                    new_row.append(0)
                else:
                    # Try to represent as fraction with denominator power of 2
                    # Since we're looking for half-integer entries, denominator should be 1 or 2
                    frac = Fraction(val).limit_denominator(2)
                    if abs(frac - val) < tol:
                        if frac.denominator == 1:
                            new_row.append(int(frac))
                        else:
                            new_row.append(frac.numerator / frac.denominator)
                    else:
                        # Try denominator 4
                        frac = Fraction(val).limit_denominator(4)
                        if abs(frac - val) < tol:
                            if frac.denominator == 1:
                                new_row.append(int(frac))
                            else:
                                new_row.append(frac.numerator / frac.denominator)
                        else:
                            # Keep as float if cannot represent exactly
                            new_row.append(val)
            result.append(new_row)
        return result
    
    return convert_matrix(U), convert_matrix(V), convert_matrix(W)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    
    # Start from the known schoolbook rank
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
    
    # Try to find lower rank decompositions
    # Known lower bounds and conjectures for 2x3 * 3x8
    # We'll try ranks from 40 down to 24
    
    for rank in range(40, 23, -1):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            np.random.seed(42 + restart * 1000 + rank)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize to avoid overflow/underflow
            U = U / np.sqrt(np.sum(U**2))
            V = V / np.sqrt(np.sum(V**2))
            W = W / np.sqrt(np.sum(W**2))
            
            # Alternating Least Squares
            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Solve for U[i*m+j, :]
                        # For each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        try:
                            U_new = np.linalg.lstsq(A, b, rcond=None)[0]
                            U[i*m+j, :] = U_new
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        try:
                            V_new = np.linalg.lstsq(A, b, rcond=None)[0]
                            V[j*p+k, :] = V_new
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        try:
                            W_new = np.linalg.lstsq(A, b, rcond=None)[0]
                            W[k*n+i, :] = W_new
                        except:
                            pass
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    converged = True
                    break
            
            if converged and _verify(U, V, W, n, m, p):
                found = True
                best_rank = rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if found:
            break
    
    # Convert to rational/half-integer if possible
    U_r, V_r, W_r = _to_rational(np.array(best_U), np.array(best_V), np.array(best_W))
    
    return {"U": U_r, "V": V_r, "W": W_r}
