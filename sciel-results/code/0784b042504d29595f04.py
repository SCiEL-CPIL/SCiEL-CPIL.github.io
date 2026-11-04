"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 135.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    # Build numpy arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The multiplication tensor: T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each (i,j,k), check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(actual - expected) > tol:
                    return False
                # Also check orthogonality: for different (i',j',k') with at least one index different,
                # the product should be 0 if it's a proper decomposition (not strictly required for correctness)
    # Also check that for non-matching indices, the sum is 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i, j, k) == (i2, j2, k2):
                                continue
                            actual = 0.0
                            for c in range(rank):
                                actual += U_np[i * m + j, c] * V_np[j2 * p + k2, c] * W_np[k2 * n + i2, c]
                            if abs(actual) > tol:
                                return False
    return True


def _try_rank(n, m, p, target_rank, max_iters=200, tol=1e-8):
    """
    Try to find a decomposition of the given rank using iterative refinement.
    Uses a random initialization and gradient-free local search.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    for attempt in range(3):
        if time.time() > DEADLINE - 5:
            break
        
        # Initialize with random vectors
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # Normalize columns to help convergence
        for c in range(target_rank):
            norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
            if norm > 1e-12:
                U[:, c] /= norm ** (1/3)
                V[:, c] /= norm ** (1/3)
                W[:, c] /= norm ** (1/3)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            
            # Compute loss: sum over all (i,j,k,i2,j2,k2) of (sum_c U[i,j,c]*V[j2,k2,c]*W[k2,i2,c] - delta_{i,i2}delta_{j,j2}delta_{k,k2})^2
            # This is expensive, so use a sampled version
            loss = 0.0
            n_samples = min(500, n * m * p * 4)
            for _ in range(n_samples):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                i2 = random.randint(0, n-1)
                j2 = random.randint(0, m-1)
                k2 = random.randint(0, p-1)
                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                actual = np.sum(U[i*m+j, :] * V[j2*p+k2, :] * W[k2*n+i2, :])
                loss += (actual - expected) ** 2
            loss /= n_samples
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < tol:
                break
            
            # Simple local perturbation
            lr = 0.01 * (1.0 + best_loss)
            mask = np.random.rand(target_rank) < 0.3
            U[:, mask] += lr * np.random.randn(nm, mask.sum())
            V[:, mask] += lr * np.random.randn(mp, mask.sum())
            W[:, mask] += lr * np.random.randn(pn, mask.sum())
        
        if best_loss < 1e-6:
            # Convert to list
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list, "rank": target_rank}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # The theoretical minimum rank for (n,m,p) matrix multiplication is at least max(n*m, m*p, p*n)
    # and at most n*m*p. For (3,3,6): lower bound = max(9, 18, 18) = 18, upper bound = 54.
    # Known results: rank of (3,3,3) is 23 (Strassen-like). 
    # For (3,3,6), we can view it as 6 copies of (3,3,1) tensor product structure.
    # The (3,3,1) tensor (matrix-vector mult) has rank 9 (schoolbook) and it's known to be 9.
    # So (3,3,6) as direct sum of 6 (3,3,1) has rank 54.
    # But there might be savings by sharing across the k dimension.
    
    # Try to find a better decomposition
    # Start from just below schoolbook rank
    for target_rank in range(schoolbook_rank - 1, 17, -1):
        if time.time() > DEADLINE - 10:
            break
        result = _try_rank(n, m, p, target_rank, max_iters=50)
        if result is not None:
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fall back to schoolbook
    sb = _schoolbook(n, m, p)
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}

# EVOLVE-BLOCK-END
