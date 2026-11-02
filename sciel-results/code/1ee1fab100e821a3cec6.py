"""Tensor decomposition for 2x3 x 3x9 matrix multiplication using numerical optimization search."""
import time
import numpy as np
import sys

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # T[i, j, k] = 1 for all i, j, k in the matmul tensor
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(lhs - rhs) > 1e-5:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _optimize_rank(n, m, p, target_rank, deadline, seed=0):
    """
    Try to find a decomposition of rank target_rank using random initialization 
    and gradient descent on the tensor fitting error.
    Returns (U, V, W) if found and verified, else None.
    """
    np.random.seed(seed)
    
    # Initialize factors with small random values
    U = np.random.randn(n*m, target_rank) * 0.5
    V = np.random.randn(m*p, target_rank) * 0.5
    W = np.random.randn(p*n, target_rank) * 0.5
    
    # We want to minimize E = sum_{i,j,k} (1 - sum_r U[i,j,r] V[j,k,r] W[k,i,r])^2
    # Let's use a simple gradient descent or Newton-like method.
    # Since dimensions are small, we can compute gradients numerically or analytically.
    
    # Analytical gradient for U[i,j,r]:
    # dE/du = -2 * sum_{i,j,k} (1 - sum_r' ...) * V[j,k,r] * W[k,i,r]
    # This is expensive to compute fully each step, but n,m,p are tiny.
    
    start_time = time.time()
    lr = 0.01
    # Adaptive learning rate
    for step in range(2000):
        if time.time() > deadline:
            break
            
        # Compute current tensor T_decomp
        # T_decomp[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Residual R[i, j, k] = 1 - T_decomp[i, j, k]
        
        # Efficient computation using einsum
        # U shape: (n*m, R) -> reshape to (n, m, R)
        U_reshaped = U.reshape(n, m, target_rank)
        V_reshaped = V.reshape(m, p, target_rank)
        W_reshaped = W.reshape(p, n, target_rank)
        
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        R_res = 1.0 - T
        
        # Check convergence
        err = np.sum(R_res**2)
        if err < 1e-8:
            # Snap to nearest integers/half-integers if close? 
            # The problem allows integer/half-integer. 
            # If the values are close to such, we can try rounding.
            # But first, let's just return if verified.
            # We need to convert back to list of lists for the interface
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Verify
            if _verify(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
            
            # If not verified due to floating point, try rounding to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            U_list_r = U_rounded.tolist()
            V_list_r = V_rounded.tolist()
            W_list_r = W_rounded.tolist()
            
            if _verify(U_list_r, V_list_r, W_list_r, n, m, p):
                return U_list_r, V_list_r, W_list_r
            continue

        # Compute gradients
        # dE/du[i, j, r] = -2 * sum_{k} R_res[i, j, k] * V[j, k, r] * W[k, i, r]
        # dE/dv[j, k, r] = -2 * sum_{i} R_res[i, j, k] * U[i, j, r] * W[k, i, r]
        # dE/dw[k, i, r] = -2 * sum_{j} R_res[i, j, k] * U[i, j, r] * V[j, k, r]
        
        # Using einsum for gradients
        # dU_reshaped[i, j, r] = -2 * sum_k R_res[i, j, k] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
        dU = -2.0 * np.einsum('ijk,jkr,kir->ijr', R_res, V_reshaped, W_reshaped)
        dV = -2.0 * np.einsum('ijk,ijr,kir->jkr', R_res, U_reshaped, W_reshaped)
        dW = -2.0 * np.einsum('ijk,ijr,jkr->kir', R_res, U_reshaped, V_reshaped)
        
        # Update
        U -= lr * dU.reshape(n*m, target_rank)
        V -= lr * dV.reshape(m*p, target_rank)
        W -= lr * dW.reshape(p*n, target_rank)
        
        # Decay learning rate
        if step % 100 == 0 and step > 0:
            lr *= 0.95
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Try schoolbook (rank 54) as baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # 2. Try to find lower rank decompositions using optimization
    # We know R(2,3,3)=14, so R(2,3,9) <= 42.
    # Let's try ranks: 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2
    # But optimization might get stuck in local minima.
    # We will try a few seeds for each rank.
    
    ranks_to_try = [42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]
    
    # Time budget per rank attempt
    # Total time available ~140s.
    # Let's allocate more time to promising lower ranks? Or just iterate.
    # Since 42 is the theoretical upper bound from block 14s, it's a good target.
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        # Try a few seeds
        seeds = [0, 1, 2, 3, 4]
        for seed in seeds:
            if time.time() > deadline - 2:
                break
                
            res = _optimize_rank(n, m, p, rank, deadline, seed)
            if res is not None:
                U, V, W = res
                # Verify again to be absolutely sure
                if _verify(U, V, W, n, m, p):
                    if rank < best_rank:
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        # If we found a very good one, we might break early
                        if rank <= 20:
                            break
        if rank <= 20:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    res = construct()
    print(f"Found rank: {len(res['U'][0])}")
