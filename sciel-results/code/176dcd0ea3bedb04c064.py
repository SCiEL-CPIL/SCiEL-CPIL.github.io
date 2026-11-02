import numpy as np
import random
import time
from itertools import product

DIMENSIONS = [4, 4, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor: T[i,j,k] = W[k*n+i] * U[i*m+j] * V[j*p+k]
    # Actually, the decomposition says:
    # T_{ijk} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
    # where T_{ijk} is the matrix multiplication tensor (1 if i,j,k are "connected" in the matmul)
    
    # For matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 (it's a structural tensor)
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Vectorized verification."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This is like a batched dot product
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # For a fixed j: U[i*m+j, :] for all i, V[j*p+k, :] for all k, W[k*n+i, :] 
    # We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # Let's compute it as: for each (i,j,k), take the column vectors
    # U[i*m+j, :] (rank,), V[j*p+k, :] (rank,), W[k*n+i, :] (rank,)
    # Their elementwise product summed = their "triple product"
    
    # Alternative: fix j. Then for all i,k:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # This is equivalent to: (U_j * V_j^T) contracted with W appropriately
    # Actually, let's just do it with broadcasting for each j
    
    for j in range(m):
        Uj = U[np.arange(n)*m + j, :]  # (n, rank)
        Vj = V[np.arange(p)*1 + j*p, :]  # This is wrong indexing
        
        # V[j*p + k, :] for k in range(p) -> V[j*p : (j+1)*p, :]
        Vj = V[j*p:(j+1)*p, :]  # (p, rank)
        
        # W[k*n + i, :] for i in range(n), k in range(p)
        # W is (p*n, rank), indexed as W[k*n + i, :]
        # So W[:, :] reshaped as (p, n, rank)
        W_reshaped = W.reshape(p, n, rank)  # W[k*n+i, r] -> W_reshaped[k, i, r]
        
        # We need: for each (i,k), sum_r Uj[i,r] * Vj[k,r] * W_reshaped[k,i,r]
        # = sum_r (Uj[i,r] * Vj[k,r]) * W_reshaped[k,i,r]
        # = einsum('ir,kr,ki->ik', Uj, Vj, W_reshaped)
        
        result = np.einsum('ir,kr,ki->ik', Uj, Vj, W_reshaped)
        if not np.allclose(result, np.ones((n, p)), atol=1e-9):
            return False
    
    return True

def _try_rank(n, m, p, rank, max_time=120.0):
    """Try to find a decomposition with the given rank."""
    start_time = time.time()
    
    # Use ALS-like approach
    # Initialize randomly
    best_result = None
    
    for attempt in range(10):
        if time.time() - start_time > max_time * 0.9:
            break
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Scale to get reasonable magnitudes
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # ALS iterations
        converged = False
        for it in range(200):
            if time.time() - start_time > max_time * 0.9:
                break
            
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is: (U[i*m+j] * (V_j * W_i^T)) = 1
            # Where V_j is V[j*p:(j+1)*p, :] and W_i involves W[k*n+i, :]
            
            # Actually, let's use a different approach: 
            # The constraint is: for all (i,j,k), sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
            # Fix (i,j): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
            # So U[i*m+j, :] needs to be such that its dot product with 
            # [V[j*p+k,0]*W[k*n+i,0], V[j*p+k,1]*W[k*n+i,1], ...] = 1 for each k
            
            # This is a linear system for U[i*m+j, :]
            
            # For each (i,j), build the system:
            # A_{k,r} = V[j*p+k,r] * W[k*n+i,r]
            # Then A @ U[i*m+j, :] = 1 for all k
            
            for i in range(n):
                for j in range(m):
                    # Build A: (p, rank)
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve A @ x = 1
                    # Use least squares
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Similarly update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Similarly update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Check convergence
            if _verify_decomposition_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                converged = True
                break
        
        if converged:
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start from the trivial decomposition (rank = n*m*p)
    trivial_rank = n * m * p
    
    U_trivial = [[0]*trivial_rank for _ in range(n*m)]
    V_trivial = [[0]*trivial_rank for _ in range(m*p)]
    W_trivial = [[0]*trivial_rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j][col] = 1
                V_trivial[j*p+k][col] = 1
                W_trivial[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = trivial_rank
    
    # Try to find lower rank decompositions
    start_time = time.time()
    max_time = 145.0  # Leave 5 seconds margin
    
    # Try various ranks from low to high
    # For 4x4x6, the Strassen-like bounds suggest rank could be significantly lower
    # Let's try from 40 up to trivial_rank
    for rank in range(30, best_rank):
        if time.time() - start_time > max_time:
            break
        
        print(f"Trying rank {rank}...")
        result = _try_rank(n, m, p, rank, max_time=max_time - (time.time() - start_time))
        
        if result is not None:
            if _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = rank
                print(f"Found valid decomposition at rank {rank}")
                # Try even lower
                # But continue searching for lower ranks
    
    return best_result
