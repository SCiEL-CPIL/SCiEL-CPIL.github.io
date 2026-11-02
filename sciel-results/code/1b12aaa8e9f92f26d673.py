import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def _get_target_tensor():
    """
    Construct the target matrix multiplication tensor T_mm.
    T_mm[i*m + j, j*p + r, r*n + i] = 1
    """
    n, m, p = DIMENSIONS
    T_mm = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for r in range(p):
                T_mm[i*m + j, j*p + r, r*n + i] = 1.0
    return T_mm

def _verify(U, V, W, T_target, tol=1e-8):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct the tensor T from U, V, W
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    T_calc = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
    return np.allclose(T_calc, T_target, atol=tol)

def _rank11_solution():
    """
    Returns the known rank 11 decomposition.
    """
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_10_search(deadline):
    """
    Attempt to find a rank 10 decomposition using SciPy L-BFGS-B optimization.
    """
    n, m, p = DIMENSIONS
    rank = 10
    T_target = _get_target_tensor()
    
    # Dimensions
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Flatten parameters
    # x = [U_flat, V_flat, W_flat]
    
    def objective(x):
        U = x[:dim_U*rank].reshape(dim_U, rank)
        V = x[dim_U*rank:dim_U*rank + dim_V*rank].reshape(dim_V, rank)
        W = x[dim_U*rank + dim_V*rank:].reshape(dim_W, rank)
        
        # Calculate T_calc
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
        diff = T_calc - T_target
        return np.sum(diff * diff)

    def gradient(x):
        U = x[:dim_U*rank].reshape(dim_U, rank)
        V = x[dim_U*rank:dim_U*rank + dim_V*rank].reshape(dim_V, rank)
        W = x[dim_U*rank + dim_V*rank:].reshape(dim_W, rank)
        
        T_calc = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
        dT = T_calc - T_target
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(rank):
            # grad_U[:, r] = sum_{j,k} dT[:, j, k] * V[j, r] * W[k, r]
            # We can compute this efficiently with einsum
            # dT is (n*m, m*p, p*n)
            # V[:, r] is (m*p, )
            # W[:, r] is (p*n, )
            
            # grad_U[:, r] = np.einsum('ijk,j,k->i', dT, V[:, r], W[:, r])
            # grad_V[:, r] = np.einsum('ijk,i,k->j', dT, U[:, r], W[:, r])
            # grad_W[:, r] = np.einsum('ijk,i,j->k', dT, U[:, r], V[:, r])
            
            grad_U[:, r] = np.einsum('ijk,j,k->i', dT, V[:, r], W[:, r])
            grad_V[:, r] = np.einsum('ijk,i,k->j', dT, U[:, r], W[:, r])
            grad_W[:, r] = np.einsum('ijk,i,j->k', dT, U[:, r], V[:, r])
            
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_error = float('inf')
    best_solution = None
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        U = np.random.randn(dim_U, rank)
        V = np.random.randn(dim_V, rank)
        W = np.random.randn(dim_W, rank)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Optimize
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_error:
                best_error = res.fun
                best_solution = res.x
                
                if best_error < 1e-8:
                    # Success
                    U_opt = best_solution[:dim_U*rank].reshape(dim_U, rank)
                    V_opt = best_solution[dim_U*rank:dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                    W_opt = best_solution[dim_U*rank + dim_V*rank:].reshape(dim_W, rank)
                    
                    if _verify(U_opt, V_opt, W_opt, T_target):
                        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                        
        except Exception:
            continue

    # Check best solution
    if best_solution is not None and best_error < 1e-8:
        U_opt = best_solution[:dim_U*rank].reshape(dim_U, rank)
        V_opt = best_solution[dim_U*rank:dim_U*rank + dim_V*rank].reshape(dim_V, rank)
        W_opt = best_solution[dim_U*rank + dim_V*rank:].reshape(dim_W, rank)
        
        if _verify(U_opt, V_opt, W_opt, T_target):
            return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    # Reserve 10s for fallback and return
    deadline = start_time + 140
    
    # 1. Try to find Rank 10 (structural change: active search)
    result_rank_10 = _try_rank_10_search(deadline)
    
    if result_rank_10:
        return result_rank_10
        
    # 2. Fallback to known Rank 11
    return _rank11_solution()
