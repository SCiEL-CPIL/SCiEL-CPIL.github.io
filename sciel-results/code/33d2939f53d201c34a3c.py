"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the target tensor T[i,j,k] = 1 if this is a valid multiplication term
    # T is a 3D array of shape (n, m, p) where T[i,j,k] = 1 for all i,j,k
    # The matrix multiplication tensor: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try to find a decomposition of rank R < schoolbook_rank
    # Use ALS with random initialization and rounding
    
    def build_target():
        """Return the target tensor as a 3D numpy array of shape (n*m, m*p, p*n) flattened appropriately"""
        # Actually, the tensor T has indices (i,j,k) and value 1
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # For the matrix multiplication tensor, T[i,j,k] = 1 for all valid (i,j,k)
        # But the mapping is: first index = i*m+j, second = j*p+k, third = k*n+i
        # These are linked: given (i,j,k), we get specific positions in U, V, W
        # The tensor is NOT a simple 3D array with all entries 1; it's structured
        # T[a, b, c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i, else 0
        # But actually for the matmul tensor, every (i,j,k) triple contributes, and the
        # "tensor" in the decomposition sense has T[i,j,k]=1 for all i,j,k
        # The issue is that U, V, W indices are linked through i,j,k
        
        # Let me think differently. The matmul tensor is:
        # T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # We want: T_{ijk} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
        # where U is reshaped to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        
        return None  # We'll handle it differently
    
    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the matmul tensor"""
        rank = U.shape[1]
        U3 = np.array(U).reshape(n, m, rank)
        V3 = np.array(V).reshape(m, p, rank)
        W3 = np.array(W).reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def round_to_half_int(arr):
        """Round to nearest half-integer"""
        return np.round(arr * 2) / 2
    
    def try_als(rank, max_iter=500, seed=42):
        """Try ALS to find a rank-decomposition"""
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Reshape for ALS
        # T_{ijk} = 1 for all i,j,k
        # For ALS, we fix two matrices and solve for the third
        # Given V and W, solve for U: U_{ijr} such that sum_r U_{ijr} V_{jkr} W_{kir} = 1
        # For fixed j,k: sum_r U_{ijr} * (V_{jkr} * W_{kir}) = 1
        # This is: U_{ij} @ diag(V_{jk} * W_{ki}) ... no, it's element-wise
        
        # Actually, for fixed (j,k), the equation is:
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        # Let M_{r} = V[j,k,r] * W[k,i,r] for fixed (i,j,k)
        # Then U[i,j,:] dot M = 1
        
        # This is underdetermined in general. ALS works by fixing V,W and solving for U
        # via least squares: for each (i,j), we have equations for all k:
        # For each k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        # This is a linear system in U[i,j,:]: A_{k,r} = V[j,k,r] * W[k,i,r], b_k = 1
        # Solve: A @ U[i,j,:] = b where A is p x rank
        
        # Similarly for V and W.
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve A @ U[i,j,:] = 1
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V3[j, k, :] * W3[k, i, :]
                    # Solve least squares: A @ x = 1
                    try:
                        U3[i, j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    except:
                        pass
            
            # Update V: for each (j,k), solve B @ V[j,k,:] = 1
            # For fixed (j,k): sum_r V[j,k,r] * U[i,j,r] * W[k,i,r] = 1 for all i
            # B_{i,r} = U[i,j,r] * W[k,i,r], b_i = 1
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U3[i, j, :] * W3[k, i, :]
                    try:
                        V3[j, k, :], _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    except:
                        pass
            
            # Update W: for each (k,i), solve C @ W[k,i,:] = 1
            # For fixed (k,i): sum_r W[k,i,r] * U[i,j,r] * V[j,k,r] = 1 for all j
            # C_{j,r} = U[i,j,r] * V[j,k,r], b_j = 1
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, rank))
                    for j in range(m):
                        C[j, :] = U3[i, j, :] * V3[j, k, :]
                    try:
                        W3[k, i, :], _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                        err += (val - 1.0) ** 2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U = U3.copy()
                best_V = V3.copy()
                best_W = W3.copy()
            
            if err < 1e-10:
                break
        
        # Round to half-integers and verify
        if best_U is not None:
            U_rounded = round_to_half_int(best_U).reshape(n * m, rank)
            V_rounded = round_to_half_int(best_V).reshape(m * p, rank)
            W_rounded = round_to_half_int(best_W).reshape(p * n, rank)
            
            if verify_decomposition(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded, rank
        
        return None
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But actually the rank must be at least the max of the matrix dimensions of the factors
    # For matmul, rank >= max(n*m, m*p, p*n) is not quite right; the rank is the tensor rank
    # Let's try from 30 upward
    
    best_result = None
    for rank in range(30, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        for seed in range(5):
            if time.time() > deadline:
                break
            result = try_als(rank, max_iter=200, seed=seed + rank * 100)
            if result is not None:
                U, V, W, r = result
                if best_result is None or r < best_result[3]:
                    best_result = (U, V, W, r)
                break  # Found a valid one for this rank
        
        if best_result is not None:
            break
    
    # If no better decomposition found, return schoolbook
    if best_result is not None:
        U, V, W, r = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Schoolbook fallback
    rank = schoolbook_rank
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

# EVOLVE-BLOCK-END
