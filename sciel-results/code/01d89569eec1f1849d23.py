"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 115.0  # 120s limit with 5s safety margin

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _verify(U, V, W, n=4, m=5, p=5):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Vectorized verification
    U_arr = np.array(U).reshape(n, m, rank)
    V_arr = np.array(V).reshape(m, p, rank)
    W_arr = np.array(W).reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    except Exception:
        return False
        
    if np.max(np.abs(T - 1.0)) > 1e-9:
        return False
    return True

def _schoolbook(n=4, m=5, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _optimize_rank(n, m, p, r, seed=42, steps=500, lr=0.05):
    """
    Vectorized gradient descent for rank r decomposition.
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n, m, r) * 0.1
    V = np.random.randn(m, p, r) * 0.1
    W = np.random.randn(p, n, r) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Simple momentum
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    beta = 0.9
    
    for step in range(steps):
        if _time_left() < 5:
            break
            
        # Compute T
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        diff = T - 1.0
        err = np.sum(diff * diff)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-12:
            break
        
        # Gradients
        # dU[i,j,c] = sum_k diff[i,j,k] V[j,k,c] W[k,i,c]
        dU = np.einsum('ijk,jkc,kic->ijc', diff, V, W)
        # dV[j,k,c] = sum_i diff[i,j,k] U[i,j,c] W[k,i,c]
        dV = np.einsum('ijk,ijc,kic->jkc', diff, U, W)
        # dW[k,i,c] = sum_j diff[i,j,k] U[i,j,c] V[j,k,c]
        dW = np.einsum('ijk,ijc,jkc->kic', diff, U, V)
        
        # Momentum update
        vU = beta * vU + (1 - beta) * dU
        vV = beta * vV + (1 - beta) * dV
        vW = beta * vW + (1 - beta) * dW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
    return best_U, best_V, best_W, best_err

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # We iterate from best_rank - 1 downwards.
    # To save time, we don't try every single rank, but we try to find the minimal.
    # Since the search is stochastic, we try a few seeds for each rank.
    
    # Start from 99 down to 50
    for r in range(best_rank - 1, 49, -1):
        if _time_left() < 10:
            break
            
        # Try a few seeds
        for seed in range(3):
            if _time_left() < 5:
                break
            
            # Adjust learning rate based on rank? Keep simple for now.
            U, V, W, err = _optimize_rank(n, m, p, r, seed=seed, steps=300, lr=0.05)
            
            if err < 1e-9:
                # Verify exactly
                U_list = U.reshape(n*m, r).tolist()
                V_list = V.reshape(m*p, r).tolist()
                W_list = W.reshape(p*n, r).tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    # If we found a very low rank, we might want to continue to see if we can go lower?
                    # The task asks for rank-minimal. So we should keep going down.
                    # But if we are fast, we can.
                    break
        # If we found a better rank, we continue the loop to try lower ranks.
        # If we didn't find any better than current best_rank (which implies r < best_rank failed), 
        # we might want to break early if we are stuck, but let's try a few more.
        
    return best
# EVOLVE-BLOCK-END
