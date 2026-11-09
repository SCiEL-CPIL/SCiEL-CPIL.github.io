"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 24
    P = p * n  # 24

    def schoolbook_rank():
        return n * m * p  # 72

    # Build the target multiplication tensor T[i,j,k] = 1 for all i,j,k
    # T is a 3D array of shape (n, m, p)
    T = np.ones((n, m, p))

    def verify_decomposition(U, V, W, tol=1e-9):
        """
        Verify that U, V, W exactly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        The tensor entry T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        """
        rank = U.shape[1]
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - T[i, j, k]) > tol:
                        return False
        return True

    def random_decomposition(rank, seed=None):
        """Generate a random decomposition and try to optimize it."""
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(N, rank) * 0.5
        V = np.random.randn(M, rank) * 0.5
        W = np.random.randn(P, rank) * 0.5
        
        return U, V, W

    def optimize_decomposition(U, V, W, max_iter=100, tol=1e-12):
        """
        Optimize the decomposition using alternating least squares.
        For each factor, fix the others and solve the linear system.
        """
        rank = U.shape[1]
        
        for iteration in range(max_iter):
            # Update U
            for i in range(n):
                for j in range(m):
                    # T[i,j,:] = sum_r U[i*m+j, r] * V[j*p+:, r] * W[:, r]
                    # For fixed i,j: for each k, T[i,j,k] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                    # This is a linear system in U[i*m+j, :]
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # b[k] = T[i,j,k]
                    A = np.zeros((p, rank))
                    b = np.zeros(p)
                    for k in range(p):
                        for r in range(rank):
                            A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        b[k] = T[i, j, k]
                    # Solve A @ U_row = b
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    # T[:,j,k] = sum_r U[:, r] * V[j*p+k, r] * W[k*n+:, r]
                    # For fixed j,k: for each i, T[i,j,k] = sum_r (U[i*m+:, r] * W[k*n+i, r]) * V[j*p+k, r]
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # b[i] = T[i,j,k]
                    A = np.zeros((n, rank))
                    b = np.zeros(n)
                    for i in range(n):
                        for r in range(rank):
                            A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        b[i] = T[i, j, k]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    # T[i,:,k] = sum_r U[i*m+:, r] * V[:, r] * W[k*n+i, r]
                    # For fixed k,i: for each j, T[i,j,k] = sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # b[j] = T[i,j,k]
                    A = np.zeros((m, rank))
                    b = np.zeros(m)
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        b[j] = T[i, j, k]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass

            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error = max(error, abs(val - T[i, j, k]))
            
            if error < tol:
                break

        return U, V, W

    def check_error(U, V, W):
        """Compute the maximum error of the decomposition."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - T[i, j, k]))
        return max_err

    # Start with the schoolbook decomposition as a baseline
    best_U = np.zeros((N, schoolbook_rank()))
    best_V = np.zeros((M, schoolbook_rank()))
    best_W = np.zeros((P, schoolbook_rank()))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    best_rank = schoolbook_rank()

    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But for matrix multiplication, the rank is at least max(n, m, p) * min(n, m, p) in some cases
    # Actually, for multiplying n x m by m x p, the tensor rank is at least max(n*m, m*p, p*n) / something
    # Let's try ranks from 24 upwards

    for rank in range(24, best_rank + 1):
        if time.time() > deadline - 30:  # Leave 30s safety margin
            break

        # Try multiple random initializations
        found = False
        for attempt in range(10):
            if time.time() > deadline - 30:
                break

            U, V, W = random_decomposition(rank, seed=rank * 100 + attempt)
            U, V, W = optimize_decomposition(U, V, W, max_iter=50)

            err = check_error(U, V, W)
            if err < 1e-9:
                # Found a valid decomposition!
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = rank
                found = True
                break

        if found:
            # Try to go even lower
            continue

    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < schoolbook_rank():
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
