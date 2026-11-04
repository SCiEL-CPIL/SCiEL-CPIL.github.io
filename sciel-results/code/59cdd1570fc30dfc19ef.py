"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def make_schoolbook():
        """Return the trivial (schoolbook) decomposition with rank = n*m*p = 54."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank_numerical(rank, max_iters=20000):
        """Try to find a valid decomposition of given rank using numerical optimization."""
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(42)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Target: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        target = np.ones((n, m, p))
        
        def compute_residual(U, V, W):
            residual = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] = np.dot(np.multiply(U[i*m+j], V[j*p+k]), W[k*n+i])
            return residual
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        lr = 0.5
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute gradient-like update using random perturbation
            U_pert = U + 0.1 * np.random.randn(n * m, rank)
            V_pert = V + 0.1 * np.random.randn(m * p, rank)
            W_pert = W + 0.1 * np.random.randn(p * n, rank)
            
            res_pert = compute_residual(U_pert, V_pert, W_pert)
            err_pert = np.sum((res_pert - target) ** 2)
            
            res_curr = compute_residual(U, V, W)
            err_curr = np.sum((res_curr - target) ** 2)
            
            if err_pert < err_curr:
                U, V, W = U_pert.copy(), V_pert.copy(), W_pert.copy()
                if err_pert < best_err:
                    best_err = err_pert
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err_pert < 1e-8:
                        break
                # Decrease learning rate if stuck
                lr *= 0.99
            else:
                lr *= 0.95
            
            if err_pert < 1e-6:
                break
        
        if best_err < 1e-5:
            # Try to round to integers or half-integers
            for scale in [1, 2, 4]:
                U_round = np.round(best_U * scale) / scale
                V_round = np.round(best_V * scale) / scale
                W_round = np.round(best_W * scale) / scale
                
                res = compute_residual(U_round, V_round, W_round)
                if np.allclose(res, target, atol=1e-6):
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
            
            # Try integer rounding
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            res = compute_residual(U_int, V_int, W_int)
            if np.allclose(res, target, atol=1e-6):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None
    
    # Try to construct a rank-46 solution using two rank-23 decompositions for 3x3x3
    # First, let's try numerical optimization for various ranks
    best_result = None
    
    for rank in [46, 45, 44, 43, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 23]:
        if time.time() > deadline - 30:
            break
        result = try_rank_numerical(rank, max_iters=15000)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
