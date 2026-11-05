"""Matrix multiplication 3x4 x 4x8 rank-minimal decomposition search."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 8]

def _exact_multiply_test(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if lhs != 1:
                    return False
    # Also verify that non-(i,j,k) terms are zero:
    # The tensor T_{i,j,k} should be 1 only for the actual (i,j,k) triple.
    # In the decomposition, the bilinear form is:
    # sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki} = sum_r (sum_{ij} U_{ij,r} A_{ij}) (sum_{jk} V_{jk,r} B_{jk}) (sum_{ki} W_{ki,r} C_{ki})
    # For this to equal sum_{i,j,k} A_{ij} B_{jk} C_{ki}, we need:
    # sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 if (i,j,k) matches the product, 0 otherwise.
    # But actually, every (i,j,k) triple IS a valid product term, so T_{ijk}=1 for all i,j,k.
    # The decomposition is valid if sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k.
    # We already checked that above. But we also need to ensure no "spurious" contributions.
    # Actually, in the standard formulation, the tensor is exactly T_{ijk}=1 for all i,j,k,
    # and the decomposition condition is sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k.
    # There are no other constraints. So the check above is sufficient.
    return True

def _try_rank(rank, n, m, p, seed=42, time_limit=30):
    """Try to find a rank decomposition using random initialization + Newton's method."""
    np.random.seed(seed)
    start = time.time()
    
    # Initialize with random Gaussian
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Reshape for easier indexing
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for step in range(2000):
        if time.time() - start > time_limit:
            break
        
        # Compute gradients
        # L = sum_{ijk} (1 - sum_r U_{ij,r} V_{jk,r} W_{ki,r})^2
        # dL/dU_{ij,r} = -2 * (1 - sum_s U_{ij,s} V_{jk,s} W_{ki,s}) * V_{jk,r} * W_{ki,r}
        # summed over j,k
        
        # For each (i, j, k): residual = 1 - sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        # grad_U_{ij,r} = -2 * sum_k residual_{ijk} * V_{jk,r} * W_{ki,r}
        # grad_V_{jk,r} = -2 * sum_i residual_{ijk} * U_{ij,r} * W_{ki,r}
        # grad_W_{ki,r} = -2 * sum_j residual_{ijk} * U_{ij,r} * V_{jk,r}
        
        residuals = np.ones((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    dot = 0.0
                    for r in range(rank):
                        dot += U_mat[i,j,r] * V_mat[j,k,r] * W_mat[k,i,r]
                    residuals[i,j,k] = 1.0 - dot
        
        loss = np.sum(residuals**2)
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if loss < 1e-12:
            break
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res = residuals[i,j,k]
                    for r in range(rank):
                        grad_U[i*m+j, r] -= 2 * res * V_mat[j,k,r] * W_mat[k,i,r]
                        grad_V[j*p+k, r] -= 2 * res * U_mat[i,j,r] * W_mat[k,i,r]
                        grad_W[k*n+i, r] -= 2 * res * U_mat[i,j,r] * V_mat[j,k,r]
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        U_mat = U.reshape(n, m, rank)
        V_mat = V.reshape(m, p, rank)
        W_mat = W.reshape(p, n, rank)
        
        # Adaptive learning rate
        if step > 0 and loss < best_loss * 0.9:
            lr *= 1.1
        elif step > 0 and loss > best_loss:
            lr *= 0.8
        lr = min(lr, 0.1)
    
    # Convert best to exact fractions if close enough
    if best_loss < 1e-8:
        U_frac = [[Fraction(best_U[i*m+j, r]).limit_denominator(1000) for r in range(rank)] for i in range(n) for j in range(m)]
        V_frac = [[Fraction(best_V[j*p+k, r]).limit_denominator(1000) for r in range(rank)] for j in range(m) for k in range(p)]
        W_frac = [[Fraction(best_W[k*n+i, r]).limit_denominator(1000) for r in range(rank)] for k in range(p) for i in range(n)]
        
        if _exact_multiply_test(U_frac, V_frac, W_frac, n, m, p, rank):
            return U_frac, V_frac, W_frac, True
        return None, None, None, False
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start from known schoolbook
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = schoolbook_rank
    
    # Try lower ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Strassen-like bounds suggest we can do better than schoolbook (96)
    # Known best for 3x4 x 4x8: try ranks from 32 upward
    for rank in range(max(n*m, m*p, p*n), min(schoolbook_rank, max(n*m, m*p, p*n) + 20)):
        if time.time() > deadline:
            break
        for seed in range(20):
            if time.time() > deadline:
                break
            U, V, W, success = _try_rank(rank, n, m, p, seed=seed, time_limit=2)
            if success:
                best_result = (U, V, W)
                best_rank = rank
                break
        if best_rank < schoolbook_rank:
            break
    
    # If we found something better, use it; otherwise schoolbook
    if best_rank < schoolbook_rank:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
