"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_matrices(U, V, W):
    """
    Convert factor matrices U (n*m, r), V (m*p, r), W (p*n, r) 
    to the tensor form. The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = U.shape[0] // m, U.shape[0] // m, W.shape[0] // n
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    return T

def _als_step(U, V, W, T, n, m, p, r, max_iter=200, tol=1e-10):
    """
    ALS optimization to find rank-r decomposition of tensor T.
    Returns (U, V, W, error)
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                # T[i, :, k] = sum_j U[i*m+j, :] * V[j*p+k, :] outer W[k*n+i, :]
                # More precisely: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed i,k: T[i,:,k] is a vector over j
                # T[i,j,k] = sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r])
                # So for each j: T[i,j,k] = dot(W[k*n+i, :], U[i*m+j, :] * V[j*p+k, :])
                # This is a linear system in W[k*n+i, :]
                
                # Build system: for each j in 0..m-1:
                # sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = T[i,j,k]
                
                # Let a_j[r] = U[i*m+j,r] * V[j*p+k,r]
                # Then sum_r W[k*n+i,r] * a_j[r] = T[i,j,k] for j=0..m-1
                
                # This is: A @ W[k*n+i, :]^T = T[i,:,k]^T where A[j,r] = U[i*m+j,r]*V[j*p+k,r]
                A = np.zeros((m, r))
                b = np.zeros(m)
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b[j] = T[i, j, k]
                
                if r >= m:
                    # Overdetermined, use least squares
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = sol
                else:
                    # Underdetermined, use pseudo-inverse
                    W[k*n+i, :] = np.linalg.pinv(A) @ b
        
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                # T[i,j,:] = sum_k U[i*m+j, :] * V[j*p+k, :] outer W[k*n+i, :]
                # For fixed i,j: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Let a_k[r] = V[j*p+k,r] * W[k*n+i,r]
                # sum_r U[i*m+j,r] * a_k[r] = T[i,j,k] for k=0..p-1
                
                A = np.zeros((p, r))
                b = np.zeros(p)
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b[k] = T[i, j, k]
                
                if r >= p:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = sol
                else:
                    U[i*m+j, :] = np.linalg.pinv(A) @ b
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                # T[:,j,k] = sum_i U[i*m+j, :] * V[j*p+k, :] outer W[k*n+i, :]
                # For fixed j,k: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Let a_i[r] = U[i*m+j,r] * W[k*n+i,r]
                # sum_r V[j*p+k,r] * a_i[r] = T[i,j,k] for i=0..n-1
                
                A = np.zeros((n, r))
                b = np.zeros(n)
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b[i] = T[i, j, k]
                
                if r >= n:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = sol
                else:
                    V[j*p+k, :] = np.linalg.pinv(A) @ b
        
        # Check convergence
        T_recon = _tensor_to_matrices(U, V, W)
        error = np.max(np.abs(T_recon - T))
        if error < tol:
            break
    
    return U, V, W, error

def _snaps_to_half_integers(U, V, W, n, m, p):
    """
    Try to snap entries to half-integers and check if the decomposition is still exact.
    Returns (U_snap, V_snap, W_snap, success)
    """
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    T_recon = _tensor_to_matrices(U_snap, V_snap, W_snap)
    T_orig = _build_tensor(n, m, p)
    
    if np.max(np.abs(T_recon - T_orig)) < 1e-8:
        return U_snap, V_snap, W_snap, True
    return None, None, None, False

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x5x6 matrix multiplication.
    Uses ALS optimization starting from various initializations.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = 130  # seconds, leaving 20s margin
    
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook rank 90 as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try different ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But we need at least the rank of the tensor, which is at least max(n*m, m*p, p*n)
    # Actually the rank is at least the maximum of the three dimensions' products
    # For matmul, rank >= max(n*m, m*p, p*n) = 30 for our case
    
    # Let's try ranks from 40 to 90
    np.random.seed(42)
    
    for target_rank in range(40, rank_school + 1, 2):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(10):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            U, V, W, error = _als_step(U, V, W, T, n, m, p, target_rank, max_iter=100, tol=1e-12)
            
            if error < 1e-8:
                # Check if we can snap to half-integers
                U_s, V_s, W_s, success = _snaps_to_half_integers(U, V, W, n, m, p)
                if success:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_s, V_s, W_s
                    break  # Found a solution at this rank, move to next lower rank
                else:
                    # Use the non-snapped version but it must be exact
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
