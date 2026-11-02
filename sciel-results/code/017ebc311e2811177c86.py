"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _to_int_or_half(matrix):
    """Convert a numpy matrix to a list of lists with integer or half-integer entries if close."""
    # Multiply by 2 to check if all entries are multiples of 0.5
    doubled = 2.0 * matrix
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=1e-6):
        # Entries are half-integers
        result = np.round(matrix).tolist()
        # Actually, let's be precise. If doubled is integer, matrix is half-integer.
        # We want to return the values.
        # The task says "integer/half-integer entries".
        # If the matrix entries are e.g. 0.5, 1.0, -0.5, etc.
        # Let's just return the list of lists of floats that are close to half-integers.
        # Or better, return the exact half-integers if possible.
        return matrix.tolist()
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i, j, k] = 1 if i, j, k are valid indices (which they always are in the loop)
    # Wait, the tensor for matrix multiplication is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So the tensor is T[i, j, k] = 1 for all i, j, k? No.
    # The tensor is defined such that C = A*B is computed as:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In the decomposition:
    # C[i, k] = sum_c (sum_j U[i*m+j, c] A[i, j]) * (sum_k V[j*p+k, c] B[j, k]) * W[k*n+i, c] ?
    # No, the standard decomposition is:
    # C[i, k] = sum_c U[i, c] V[k, c] ... no.
    
    # Let's look at the schoolbook decomposition in the seed:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # The product sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] is 1 for the specific (i, j, k) triple.
    # So the tensor T[i, j, k] = 1 for all i, j, k?
    # Yes, the matrix multiplication tensor has T[i, j, k] = 1 for all valid i, j, k.
    # Because C[i, k] = sum_j A[i, j] B[j, k].
    # Each term A[i, j] B[j, k] is a separate component.
    # So we need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
    
    # Let's check this condition.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _try_rank(rank, n, m, p, max_time=10.0):
    """Try to find a rank-`rank` decomposition. Returns (U, V, W) or None."""
    start_time = time.time()
    # Random initialization
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # We want to minimize sum_{i,j,k} (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1)^2
    # This is a non-convex optimization problem.
    # We can use gradient descent or alternating minimization.
    # For simplicity, let's try a few random restarts with alternating minimization.
    
    best_U, best_V, best_W = None, None, None
    best_error = float('inf')
    
    for restart in range(20):
        if time.time() - start_time > max_time - 2:
            break
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        for it in range(200):
            if time.time() - start_time > max_time - 2:
                break
            # Update U: for each (i, j), U[i*m+j, :] = argmin_u sum_k (sum_c u_c V[j*p+k, c] W[k*n+i, c] - 1)^2
            # This is a linear system for u.
            # Let M_u be the matrix where M_u[c, k] = V[j*p+k, c] W[k*n+i, c]
            # Then we want to minimize ||M_u u - 1||^2, where 1 is a vector of 1s of length p.
            # So u = M_u^+ 1 (pseudo-inverse).
            for i in range(n):
                for j in range(m):
                    M_u = np.zeros((rank, p))
                    for c in range(rank):
                        for k in range(p):
                            M_u[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    # Solve M_u^T M_u u = M_u^T 1
                    A = M_u.T @ M_u
                    b = M_u.T @ np.ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.solve(A + 1e-8*np.eye(rank), b)
                    except:
                        U[i*m+j, :] = np.linalg.lstsq(M_u, np.ones(p), rcond=None)[0]
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M_v = np.zeros((rank, n))
                    for c in range(rank):
                        for i in range(n):
                            M_v[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    A = M_v.T @ M_v
                    b = M_v.T @ np.ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.solve(A + 1e-8*np.eye(rank), b)
                    except:
                        V[j*p+k, :] = np.linalg.lstsq(M_v, np.ones(n), rcond=None)[0]
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M_w = np.zeros((rank, m))
                    for c in range(rank):
                        for j in range(m):
                            M_w[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    A = M_w.T @ M_w
                    b = M_w.T @ np.ones(m)
                    try:
                        W[k*n+i, :] = np.linalg.solve(A + 1e-8*np.eye(rank), b)
                    except:
                        W[k*n+i, :] = np.linalg.lstsq(M_w, np.ones(m), rcond=None)[0]
            
            # Compute error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (val - 1.0)**2
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            if error < 1e-6:
                break
        
        if best_error < 1e-6:
            break
    
    if best_error < 1e-6:
        # Convert to list of lists
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        # Check if entries are close to half-integers
        # For now, just return the float lists. The evaluator might accept floats.
        # But the task says "integer/half-integer entries".
        # Let's try to round to nearest half-integer and verify.
        U_half = np.round(2 * best_U) / 2.0
        V_half = np.round(2 * best_V) / 2.0
        W_half = np.round(2 * best_W) / 2.0
        if _verify_decomposition(U_half.tolist(), V_half.tolist(), W_half.tolist(), n, m, p):
            return U_half.tolist(), V_half.tolist(), W_half.tolist()
        else:
            # If rounding fails, maybe the original floats are acceptable?
            # The task says "integer/half-integer entries that exactly decompose".
            # So we must return half-integers.
            # If we can't find half-integers, we fail.
            return None
    return None

# EVOLVE-BLOCK-END
