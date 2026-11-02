"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _try_rank_am(n, m, p, rank, max_iter=300, tol=1e-8, seed=None, init_type='random'):
    """Try to find a decomposition of given rank using alternating minimization."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize
    if init_type == 'structured':
        # Use block structure: split m=4 into 2+2, p=6 into 3+3
        # This gives 4 sub-problems of 3x2x3
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Assign rank slots to blocks
        block_rank = rank // 4
        for bi in range(2):  # m blocks
            for bk in range(2):  # p blocks
                base = (bi * 2 + bk) * block_rank
                for i in range(n):
                    for j in range(2):
                        jj = bi * 2 + j
                        for k in range(3):
                            kk = bk * 3 + k
                            # Add random perturbation within block
                            for r in range(block_rank):
                                if r < 0:
                                    continue
                                U[i * m + jj, base + r] = np.random.randn() * 0.1
                                V[jj * p + kk, base + r] = np.random.randn() * 0.1
                                W[kk * n + i, base + r] = np.random.randn() * 0.1
    else:
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
    
    best_obj = float('inf')
    best_state = None
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :] given V, W
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :] given U, W
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :] given U, V
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Compute objective: sum of squared errors
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    obj += (s - 1.0) ** 2
        
        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())
        
        if obj < tol:
            break
    
    if best_obj < tol:
        U, V, W = best_state
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = max(12, 24, 18) = 24
    # But realistically, we expect rank around 32-40 for 3x4x6
    
    for rank in range(32, best_rank + 1):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Try multiple seeds and initialization types
        for seed in range(8):
            if time.time() > _DEADLINE - 5.0:
                break
            
            # Try both random and structured initialization
            for init_type in ['random', 'structured']:
                if time.time() > _DEADLINE - 5.0:
                    break
                
                result = _try_rank_am(n, m, p, rank, max_iter=150, tol=1e-8, seed=seed, init_type=init_type)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W, n, m, p, tol=1e-6):
                        best_result = (U, V, W)
                        best_rank = rank
                        break
            
            if best_result is not None:
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
