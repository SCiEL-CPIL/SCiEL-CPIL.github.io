"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=5, m=5, p=5, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check shapes
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with identity tensor T[i,j,k] = 1 if i==j==k... no, 
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 (the multiplication tensor has all entries = 1)
    # Actually the matrix multiplication tensor T_{n,n,n}[i,j,k] = 1 for all i,j,k
    
    # Reshape U, V, W to rank-3 tensors
    U3 = U_arr.reshape(n, m, rank)  # U[i,j,r]
    V3 = V_arr.reshape(m, p, rank)  # V[j,k,r]
    W3 = W_arr.reshape(p, n, rank)  # W[k,i,r]
    
    # Compute tensor product: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # The target is all ones
    target = np.ones((n, m, p))
    
    if not np.allclose(T, target, atol=tol):
        return False
    return True

def _refine(U, V, W, n=5, m=5, p=5, max_iter=2000, tol=1e-10):
    """Iteratively refine a rank-r decomposition to minimize error."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    target = np.ones((n, m, p))
    
    for it in range(max_iter):
        # Update U
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For fixed j,k: target[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
        # U[i,j,:] = (sum over k of target[i,j,k] * V[j,k,:] * W[k,i,:]) / (sum over k of (V[j,k,:] * W[k,i,:])^2)
        
        new_U3 = np.zeros_like(U3)
        new_V3 = np.zeros_like(V3)
        new_W3 = np.zeros_like(W3)
        
        for i in range(n):
            for j in range(m):
                # For U[i,j,:]:
                # We want to minimize sum_k (target[i,j,k] - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r])^2
                # Linear in U[i,j,:]:
                # Let b_k = V[j,k,:] * W[k,i,:]  (vector of length rank)
                # Then T[i,j,k] = dot(U[i,j,:], b_k)
                # Minimize sum_k (target[i,j,k] - dot(U[i,j,:], b_k))^2
                # Solution: U[i,j,:] = (sum_k target[i,j,k] * b_k) / (sum_k b_k^2)
                
                b = V3[j, :, :] * W3[:, i, :]  # shape (p, rank)
                rhs = np.zeros(rank)
                denom = 0.0
                for k in range(p):
                    rhs += target[i, j, k] * b[k, :]
                    denom += np.dot(b[k, :], b[k, :])
                
                if denom > tol:
                    new_U3[i, j, :] = rhs / denom
                else:
                    new_U3[i, j, :] = U3[i, j, :]
        
        for j in range(m):
            for k in range(p):
                # For V[j,k,:]:
                a = U3[:, j, :] * W3[k, :, :]  # shape (n, rank)
                rhs = np.zeros(rank)
                denom = 0.0
                for i in range(n):
                    rhs += target[i, j, k] * a[i, :]
                    denom += np.dot(a[i, :], a[i, :])
                
                if denom > tol:
                    new_V3[j, k, :] = rhs / denom
                else:
                    new_V3[j, k, :] = V3[j, k, :]
        
        for k in range(p):
            for i in range(n):
                # For W[k,i,:]:
                c = U3[i, :, :] * V3[:, k, :]  # shape (m, rank)
                rhs = np.zeros(rank)
                denom = 0.0
                for j in range(m):
                    rhs += target[i, j, k] * c[j, :]
                    denom += np.dot(c[j, :], c[j, :])
                
                if denom > tol:
                    new_W3[k, i, :] = rhs / denom
                else:
                    new_W3[k, i, :] = W3[k, i, :]
        
        # Compute error
        T = np.einsum('ijr,jkr,kir->ijk', new_U3, new_V3, new_W3)
        err = np.sum((T - target)**2)
        
        if err < tol:
            U3 = new_U3
            V3 = new_V3
            W3 = new_W3
            break
        
        U3 = new_U3
        V3 = new_V3
        W3 = new_W3
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    err = np.sum((T - target)**2)
    
    return U3.reshape(n*m, rank).tolist(), V3.reshape(m*p, rank).tolist(), W3.reshape(p*n, rank).tolist(), err

def _try_rank(rank, n=5, m=5, p=5, n_restarts=5, max_iter=1000, seed=42):
    """Try to find a valid decomposition of given rank."""
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_result = None
    
    for restart in range(n_restarts):
        # Random initialization with small values
        U_init = rng.randn(n*m, rank) * 0.1
        V_init = rng.randn(m*p, rank) * 0.1
        W_init = rng.randn(p*n, rank) * 0.1
        
        U_list, V_list, W_list, err = _refine(U_init, V_init, W_init, n, m, p, max_iter)
        
        if err < best_err:
            best_err = err
            best_result = (U_list, V_list, W_list)
        
        if best_err < 1e-9:
            break
    
    if best_err < 1e-9:
        return best_result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, try some known good ranks
    # Theoretical lower bound for 5x5 matrix mult is around 84
    # Best known upper bounds are around 100-110
    # Let's try from a reasonable starting point
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try ranks from 90 upward (conservative estimate of best known)
    # Use time budget to determine how many to try
    for rank in range(90, rank_school + 1, 5):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank(rank, n, m, p, n_restarts=3, max_iter=500, seed=rank)
        
        if result is not None:
            best_rank = rank
            best_result = {"U": result[0], "V": result[1], "W": result[2]}
            break  # Found a valid one, this is the minimum we found
    
    return best_result
# EVOLVE-BLOCK-END
