"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Should equal 1 for all i,j,k
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_random(rank, n, m, p, rng, max_attempts=200, density=0.15):
    """Try to find a valid decomposition of given rank using random initialization."""
    for _ in range(max_attempts):
        U = rng.standard_normal((n * m, rank))
        V = rng.standard_normal((m * p, rank))
        W = rng.standard_normal((p * n, rank))
        
        # Apply sparsity mask
        U *= (rng.random((n * m, rank)) < density)
        V *= (rng.random((m * p, rank)) < density)
        W *= (rng.random((p * n, rank)) < density)
        
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_structured(rank, n, m, p, rng):
    """
    Try structured decompositions based on Kronecker-like products.
    For 4x4x7, we can think of it as 7 independent 4x4x1 multiplications.
    Each 4x4x1 has rank at most 16 (schoolbook).
    Total rank 112 is schoolbook.
    
    Alternative: use the fact that 4 = 2*2, so 4x4 multiplication can be done
    in rank less than 16? Actually, 4x4x4 has known rank 49 (Strassen-like).
    But here it's 4x4x1 for each column, so it's matrix-vector multiplication.
    Matrix-vector 4x4 has rank 16 (each output depends on all 4 inputs).
    
    Let's try a different structured approach: exploit the tensor structure.
    """
    # For now, return None and rely on random search
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 7 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 120s limit
    
    # Start with schoolbook as guaranteed valid solution
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p  # 112
    
    rng = np.random.default_rng(42)
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But for matrix multiplication, the rank is at least n*m*p / min(n,m,p) in some cases
    # Actually, for (n x m) * (m x p), the rank is at least max(n*m, m*p, p*n)
    
    # Try ranks from best_rank-1 down to a reasonable lower bound
    min_rank_to_try = max(28, best_rank - 50)
    
    for rank in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Try random search with this rank
        result = _try_random(rank, n, m, p, rng, max_attempts=100)
        if result is not None:
            U, V, W = result
            best_rank = rank
            continue  # Try to go even lower
        
        # Try structured approach
        result = _try_structured(rank, n, m, p, rng)
        if result is not None:
            U, V, W = result
            best_rank = rank
            continue
    
    return {"U": U, "V": W, "W": W} if False else {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
