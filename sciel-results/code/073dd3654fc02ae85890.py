import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Use alternating least squares (ALS) to refine a rank-r decomposition,
    starting from the schoolbook rank-80 baseline and attempting to reduce rank
    by finding a lower-rank approximation that can be rounded to exact integers/half-integers.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill

    # Helper: Verify if a decomposition is exact
    def verify_decomposition(U, V, W):
        """
        Check if U, V, W exactly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        The tensor T[i,j,k] = 1 if i,j,k are valid indices, else 0.
        Actually, for matrix multiplication C = A*B:
        C[i,k] = sum_j A[i,j] * B[j,k]
        In tensor form: T_{i,j,k} where the bilinear form is sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        Wait, the standard tensor for matrix multiplication M_{n,m,p} has entries T[i,j,k] = 1 for all i,j,k.
        The decomposition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.
        """
        rank = U.shape[1]
        # Compute the reconstructed tensor
        # T_recon[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We can check this by computing the "contraction"
        # Let's compute the residual for all (i,j,k)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < 1e-9

    # Helper: Round to half-integers and verify
    def round_and_verify(U, V, W):
        """
        Round U, V, W to nearest half-integers and check if exact.
        """
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify_decomposition(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    # Initialize with schoolbook rank 80
    rank_init = n * m * p
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    # Try to reduce rank using ALS
    # We'll try ranks from 79 down to some lower bound
    best_U, best_V, best_W = U, V, W
    best_rank = rank_init

    # For each rank r, try to find a decomposition
    # Start from a subset of the schoolbook terms or random initialization
    for target_rank in range(rank_init - 1, 30, -1):
        if time.time() > deadline:
            break
            
        # Initialize with first target_rank terms of schoolbook
        U_try = U[:, :target_rank].copy()
        V_try = V[:, :target_rank].copy()
        W_try = W[:, :target_rank].copy()
        
        # Add small random perturbation to escape local minima
        U_try += np.random.normal(0, 0.1, U_try.shape)
        V_try += np.random.normal(0, 0.1, V_try.shape)
        W_try += np.random.normal(0, 0.1, W_try.shape)
        
        # ALS iterations
        converged = False
        for it in range(50):
            if time.time() > deadline:
                break
                
            # Update U: For each (i,j), minimize sum_k (1 - sum_r U[i,j,r] * V[j,k,r] * W[k,i,r])^2
            # This is a linear least squares problem for U[i,j,:]
            for i in range(n):
                for j in range(m):
                    # Build the system: X * u = b
                    # X[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # b[k] = 1
                    X = np.zeros((p, target_rank))
                    for k in range(p):
                        X[k, :] = V_try[j*p+k, :] * W_try[k*n+i, :]
                    b = np.ones(p)
                    # Solve X u = b
                    try:
                        U_try[i*m+j, :], residuals, rank, sv = np.linalg.lstsq(X, b, rcond=None)
                    except:
                        break
            if time.time() > deadline:
                break
                
            # Update V: For each (j,k), minimize sum_i (1 - sum_r U[i,j,r] * V[j,k,r] * W[k,i,r])^2
            for j in range(m):
                for k in range(p):
                    X = np.zeros((n, target_rank))
                    for i in range(n):
                        X[i, :] = U_try[i*m+j, :] * W_try[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_try[j*p+k, :], residuals, rank, sv = np.linalg.lstsq(X, b, rcond=None)
                    except:
                        break
            if time.time() > deadline:
                break
                
            # Update W: For each (k,i), minimize sum_j (1 - sum_r U[i,j,r] * V[j,k,r] * W[k,i,r])^2
            for k in range(p):
                for i in range(n):
                    X = np.zeros((m, target_rank))
                    for j in range(m):
                        X[j, :] = U_try[i*m+j, :] * V_try[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_try[k*n+i, :], residuals, rank, sv = np.linalg.lstsq(X, b, rcond=None)
                    except:
                        break
            if time.time() > deadline:
                break
        
        # Check if the solution is exact or can be rounded to exact
        result = round_and_verify(U_try, V_try, W_try)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break  # Found a valid lower rank, stop

    # If no lower rank found, return the schoolbook
    if best_rank == rank_init:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
