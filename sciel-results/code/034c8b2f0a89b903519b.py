import numpy as np
import time
import sys
from itertools import product
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for M_{3,4,6}.
    Tries to find a lower rank decomposition using optimization.
    Falls back to schoolbook rank 72 if no lower rank is found within time limit.
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Pre-compute the target tensor T
    # T[i, j, k] = 1 if (i, j, k) is a valid index tuple, else 0
    # Actually, the tensor for matrix multiplication C = A * B is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices (i, j, k) where T[i, j, k] = 1.
    # The factorization is T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
    # Note: The problem statement says W[k*n+i, r]. This implies the third index of W is i.
    # Let's check the interface:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # This matches the standard tensor definition for matrix multiplication.
    
    N, M, P = n, m, p
    NM, MP, PN = N * M, M * P, P * N
    
    # Construct target tensor T of shape (N, M, P)
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = 1.0

    def check_exact(U, V, W):
        """
        Checks if U, V, W exactly decompose T.
        U: (N, M, r), V: (M, P, r), W: (P, N, r)
        """
        r = U.shape[2]
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)

    def try_rank(r, max_trials=5, max_iters=50):
        """
        Attempts to find a rank-r decomposition.
        Returns (U, V, W) if found, else None.
        """
        if time.time() - start_time > time_limit:
            return None
            
        for trial in range(max_trials):
            if time.time() - start_time > time_limit:
                return None
                
            np.random.seed(12345 + trial)
            
            # Initialize with small random values
            U = np.random.randn(N, M, r) * 0.1
            V = np.random.randn(M, P, r) * 0.1
            W = np.random.randn(P, N, r) * 0.1
            
            size_U = N * M * r
            size_V = M * P * r
            size_W = P * N * r
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective_and_grad(x):
                U_flat = x[:size_U]
                V_flat = x[size_U:size_U+size_V]
                W_flat = x[size_U+size_V:]
                
                U = U_flat.reshape(N, M, r)
                V = V_flat.reshape(M, P, r)
                W = W_flat.reshape(P, N, r)
                
                T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                err = T_rec - T
                loss = np.sum(err**2)
                
                # Gradients
                # dL/dU[i,j,c] = 2 * sum_{k} err[i,j,k] * V[j,k,c] * W[k,i,c]
                grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
                # dL/dV[j,k,c] = 2 * sum_{i} err[i,j,k] * U[i,j,c] * W[k,i,c]
                grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
                # dL/dW[k,i,c] = 2 * sum_{j} err[i,j,k] * U[i,j,c] * V[j,k,c]
                grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return loss, grad

            try:
                res = opt.minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                                   options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-6})
            except Exception:
                continue
                
            if res.fun < 1e-8:
                # Extract solution
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U+size_V]
                W_flat = res.x[size_U+size_V:]
                
                U = U_flat.reshape(N, M, r)
                V = V_flat.reshape(M, P, r)
                W = W_flat.reshape(P, N, r)
                
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if check_exact(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
        
        return None

    # Try ranks from lower bound up to schoolbook rank - 1
    # Lower bound is max(NM, MP, PN) = max(12, 24, 18) = 24
    lower_bound = max(NM, MP, PN)
    
    # We will try a few ranks close to the lower bound first, as they are most likely to have solutions if any exist.
    # However, optimization is non-convex. We'll try a range.
    # To save time, we won't try all ranks from 24 to 71. We'll try a subset.
    # Let's try 24, 25, ..., 30, and maybe a few higher.
    # Given the time constraint, we'll just try a few.
    
    ranks_to_try = list(range(lower_bound, lower_bound + 10)) + [40, 50, 60]
    ranks_to_try = sorted(set(ranks_to_try))
    
    for r in ranks_to_try:
        if time.time() - start_time > time_limit - 10:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {
                "U": U.flatten(axis=1).tolist(),
                "V": V.flatten(axis=1).tolist(),
                "W": W.flatten(axis=1).tolist()
            }

    # Fallback to schoolbook
    U = np.zeros((NM, rank_schoolbook), dtype=np.int8)
    V = np.zeros((MP, rank_schoolbook), dtype=np.int8)
    W = np.zeros((PN, rank_schoolbook), dtype=np.int8)
    
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, c] = 1
                V[j * P + k, c] = 1
                W[k * N + i, c] = 1
                c += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
