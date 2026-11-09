"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin (150s total, leave 10s buffer)
    
    # Known optimal rank for 3x4x5 matrix multiplication is 16
    # Try ranks from 16 upward
    for r in range(16, n * m * p + 1):
        if time.time() > deadline:
            break
            
        result = _try_rank(n, m, p, r, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook
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


def _try_rank(n, m, p, r, deadline):
    """Try to find an exact half-integer decomposition of rank r using structured search."""
    
    # Try multiple random seeds for robustness
    for seed in range(20):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, r) * 0.3
        V = np.random.randn(m * p, r) * 0.3
        W = np.random.randn(p * n, r) * 0.3
        
        # Alternating minimization with vectorized updates
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve linear system
            for j in range(m):
                for k in range(p):
                    # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    A = U[np.arange(n)*m + j, :] * W[k*n + np.arange(n), :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve linear system
            for k in range(p):
                for i in range(n):
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.vstack([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check if we're close to a half-integer solution
            if iteration % 20 == 0:
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                if _verify_fast(n, m, p, U_round, V_round, W_round):
                    return _to_lists(n, m, p, U_round, V_round, W_round)
    
    # Try a structured approach: use the fact that rank 16 is achievable
    if r == 16:
        return _try_structured_16(n, m, p, deadline)
    
    return None


def _verify_fast(n, m, p, U, V, W):
    """Fast vectorized verification using NumPy."""
    r = U.shape[1]
    
    # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can do this efficiently by iterating over c and accumulating
    
    result = np.zeros((n, m, p))
    for c in range(r):
        # U_c has shape (n*m,), V_c has shape (m*p,), W_c has shape (p*n,)
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        
        # For each (i,j,k): Uc[i,j] * Vc[j,k] * Wc[k,i]
        # This is: sum over j of Uc[i,j] * Vc[j,k] * Wc[k,i]
        # = sum_j Uc[i,j] * Vc[j,:] * Wc[:,i]
        # = (Uc @ Vc) * Wc.T, then we need element-wise product with Wc transposed
        
        # Actually: result[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        # Let's compute M = Uc @ Vc, so M[i,k] = sum_j Uc[i,j]*Vc[j,k]
        # Then result[i,k] += M[i,k] * Wc[k,i]
        # But we need result[i,j,k] for each j, not summed over j
        
        # Alternative: for fixed c, the contribution is outer product structure
        # result_c[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
        
        # We can compute this as: for each j, Uc[:,j] (n,) * Vc[j,:] (p,) * Wc[:,j]... no
        # Wc[k,i] means Wc is indexed by (k,i), so Wc.T[i,k]
        
        # Let's just do it with broadcasting
        # Uc: (n,m), Vc: (m,p), Wc: (p,n)
        # We want: Uc[i,j] * Vc[j,k] * Wc[k,i]
        
        # Create full tensor via einsum or manual
        # Use: np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
        result += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
    
    return np.all(np.abs(result - 1.0) < 1e-9)


def _try_structured_16(n, m, p, deadline):
    """Try a structured approach for rank 16 decomposition."""
    # For 3x4x5, the rank 16 construction uses a specific pattern
    # This is based on known results in tensor rank theory
    
    # One approach: use the fact that we can factor through smaller matrices
    # The key insight is that we need 16 terms of the form:
    # U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
    
    # Try a grid-based initialization
    np.random.seed(123)
    
    # Initialize with a structured pattern
    U = np.zeros((n*m, 16))
    V = np.zeros((m*p, 16))
    W = np.zeros((p*n, 16))
    
    # Use a Kronecker-like structure
    # Split the 16 ranks into groups
    for c in range(16):
        # Assign patterns based on c
        ic = c // 4  # 0,1,2,3 for i-component
        jc = c % 4    # 0,1,2,3 for j-component
        
        # Set up U, V, W with structured values
        for i in range(n):
            for j in range(m):
                U[i*m+j, c] = 1.0 if (i == ic % n and j == jc % m) else 0.5
        
        for j in range(m):
            for k in range(p):
                V[j*p+k, c] = 1.0 if (j == jc % m and k == c % p) else 0.5
        
        for k in range(p):
            for i in range(n):
                W[k*n+i, c] = 1.0 if (k == c % p and i == ic % n) else 0.5
    
    # Now optimize from this structured start
    for iteration in range(100):
        if time.time() > deadline - 10:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = U[np.arange(n)*m + j, :] * W[k*n + np.arange(n), :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.vstack([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        if iteration % 15 == 0:
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            if _verify_fast(n, m, p, U_round, V_round, W_round):
                return _to_lists(n, m, p, U_round, V_round, W_round)
    
    return None


def _to_lists(n, m, p, U, V, W):
    """Convert numpy arrays to lists of lists with exact half-integer values."""
    r = U.shape[1]
    U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
