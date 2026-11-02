"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _check_decomposition(U, V, W, n, m, p):
    """Check if the decomposition satisfies all n*m*p tensor equations."""
    rank = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > 1e-6:
                    return False
    return True

def _verify(U, V, W):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(W[0])
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                lA = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                lB = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                C_computed[i, k] += W[k*n+i, r] * lA * lB
    return np.allclose(C_computed, C_expected, atol=1e-8)

def _try_rank_10_gauge(deadline):
    """Try to find a rank-10 decomposition using gauge fixing and multi-start optimization."""
    n, m, p = DIMENSIONS
    rank = 10
    
    def objective(params):
        # Reconstruct U, V, W from params
        # U is 4x10, V is 6x10, W is 6x10
        # We fix the first 4 columns of U to be identity-like?
        # Actually, let's just optimize all parameters but start with a good structure.
        
        U_flat = params[0:4*rank]
        V_flat = params[4*rank:4*rank+6*rank]
        W_flat = params[4*rank+6*rank:4*rank+6*rank+6*rank]
        
        U = U_flat.reshape((4, rank))
        V = V_flat.reshape((6, rank))
        W = W_flat.reshape((6, rank))
        
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    obj += (val - 1)**2
        return obj

    rng = np.random.default_rng(42)
    best_obj = float('inf')
    best_params = None
    
    num_starts = 100
    
    for start in range(num_starts):
        if time.time() > deadline:
            break
            
        # Initialize U with a structure that might help
        # e.g., first 4 columns are identity, rest are random small values
        U = np.zeros((4, rank))
        V = rng.standard_normal((6, rank)) * 0.1
        W = rng.standard_normal((6, rank)) * 0.1
        
        # Set first 4 columns of U to be close to identity
        # This helps with scaling
        for i in range(4):
            U[i, i] = 1.0
            # Add small noise to other elements in these columns to allow movement
            U[i, i+1:] += rng.standard_normal(rank - i - 1) * 0.1
            # Allow mixing with previous columns? No, keep it simple.
        
        # Actually, just random is often better if the optimizer is good.
        # Let's try a mix: some structured, some random.
        if start % 2 == 0:
             U = rng.standard_normal((4, rank)) * 0.1
             U[:, :4] += np.eye(4)
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            # Use L-BFGS-B
            result = minimize(objective, params0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
                
                if best_obj < 1e-8:
                    break
                    
        except Exception:
            continue
            
    if best_params is not None and best_obj < 1e-6:
        U_flat = best_params[0:4*rank]
        V_flat = best_params[4*rank:4*rank+6*rank]
        W_flat = best_params[4*rank+6*rank:4*rank+6*rank+6*rank]
        U = U_flat.reshape((4, rank))
        V = V_flat.reshape((6, rank))
        W = W_flat.reshape((6, rank))
        
        if _check_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _rank_11_solution():
    """Return the known rank-11 decomposition (Strassen + schoolbook for col 2)."""
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen part (terms 0-6) for 2x2 * 2x2 (cols 0,1 of B)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[4,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    U[0,2]=1; V[1,2]=1; V[4,2]=-1
    U[3,3]=1; V[3,3]=1; V[0,3]=-1
    U[0,4]=1; U[1,4]=1; V[4,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    U[1,6]=1; U[3,6]=-1; V[3,6]=1; V[4,6]=1
    
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1
    W[2,2]=1; W[2,4]=1
    W[1,1]=1; W[1,3]=1
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1
    
    # Schoolbook for col 2 (terms 7-10)
    U[0,7]=1; V[2,7]=1; W[4,7]=1
    U[1,8]=1; V[5,8]=1; W[4,8]=1
    U[2,9]=1; V[2,9]=1; W[5,9]=1
    U[3,10]=1; V[5,10]=1; W[5,10]=1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank 10 with gauge-fixed multi-start search
    result_10 = _try_rank_10_gauge(deadline)
    if result_10 is not None:
        U, V, W = result_10
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 11 (Strassen + schoolbook)
    U, V, W = _rank_11_solution()
    
    # Verify the decomposition
    if not _verify(U.tolist(), V.tolist(), W.tolist()):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 12
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
