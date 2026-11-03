import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find better decomposition for 2x3x3
    # Target rank 12
    target_rank = 12
    
    # Tensor for 2x3x3
    # T[i, j, k] = 1 if i,j,k valid
    n2, m2, p2 = 2, 3, 3
    T_233 = np.zeros((n2, m2, p2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T_233[i, j, k] = 1
                
    T_flat = T_233.flatten()
    
    def solve_233(rank):
        if time.time() > deadline - 5:
            return None
            
        # Variables: U (n2*rank), V (m2*rank), W (p2*rank)
        # Total vars: (n2 + m2 + p2) * rank = 8 * rank
        # For rank 12, 96 vars.
        
        # Objective: sum (T_flat - sum_r U[i,r]V[j,r]W[k,r])^2
        # We can vectorize this.
        
        def objective(params):
            U = params[0:n2*rank].reshape(n2, rank)
            V = params[n2*rank:n2*rank+m2*rank].reshape(m2, rank)
            W = params[n2*rank+m2*rank:].reshape(p2, rank)
            
            # Compute tensor
            # T_comp[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T_comp.flatten() - T_flat
            return diff
            
        # Initial guess
        # Random initialization
        np.random.seed(42)
        best_params = None
        best_err = float('inf')
        
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize with small random values
            size = (n2 + m2 + p2) * rank
            params0 = np.random.randn(size) * 0.1
            
            try:
                from scipy.optimize import least_squares
                res = least_squares(objective, params0, method='lm', max_nfev=1000)
                err = np.linalg.norm(res.fun)
                if err < best_err:
                    best_err = err
                    best_params = res.x
            except Exception:
                pass
                
        if best_params is None or best_err > 1e-8:
            return None
            
        # Extract U, V, W
        U = best_params[0:n2*rank].reshape(n2, rank)
        V = best_params[n2*rank:n2*rank+m2*rank].reshape(m2, rank)
        W = best_params[n2*rank+m2*rank:].reshape(p2, rank)
        
        # Check for half-integer snapping
        # If values are close to 0.5 or integers, snap them
        def snap_matrix(M):
            M_snap = np.round(M * 2) / 2.0
            # Verify if snapping keeps error low
            T_comp = np.einsum('ir,jr,kr->ijk', M_snap[0:n2], M_snap[n2:n2+m2], M_snap[n2+m2:])
            diff = T_comp.flatten() - T_flat
            if np.linalg.norm(diff) < 1e-6:
                return M_snap
            return M
            
        U = snap_matrix(U)
        V = snap_matrix(V)
        W = snap_matrix(W)
        
        # Final verification
        T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = T_comp.flatten() - T_flat
        if np.linalg.norm(diff) < 1e-6:
            return U, V, W
            
        return None

    # Try rank 12
    res_12 = solve_233(12)
    if res_12 is not None:
        U_233, V_233, W_233 = res_12
        # Construct 2x3x6 decomposition
        # Rank 24
        rank_236 = 24
        U_236 = np.zeros((n*m, rank_236))
        V_236 = np.zeros((m*p, rank_236))
        W_236 = np.zeros((p*n, rank_236))
        
        # First 12 cols: Block 0
        # U is same for both blocks
        U_236[:, 0:12] = U_233
        U_236[:, 12:24] = U_233
        
        # V: Block 0 uses B[0:3], Block 1 uses B[3:6]
        # V_236 rows are j*6 + k
        for j in range(m2):
            for k in range(p2):
                V_236[j*6 + k, 0:12] = V_233[j, 0:12]
                V_236[j*6 + (k+3), 12:24] = V_233[j, 0:12]
                
        # W: Block 0 uses C[0:3], Block 1 uses C[3:6]
        # W_236 rows are k*2 + i
        for k in range(p2):
            for i in range(n2):
                W_236[k*2 + i, 0:12] = W_233[k, 0:12]
                W_236[(k+3)*2 + i, 12:24] = W_233[k, 0:12]
                
        # Convert to lists
        U_list = U_236.tolist()
        V_list = V_236.tolist()
        W_list = W_236.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
        
    # If 12 fails, try 13, 14...
    for r in range(13, 18):
        if time.time() > deadline - 10:
            break
        res = solve_233(r)
        if res is not None:
            U_233, V_233, W_233 = res
            rank_236 = 2 * r
            U_236 = np.zeros((n*m, rank_236))
            V_236 = np.zeros((m*p, rank_236))
            W_236 = np.zeros((p*n, rank_236))
            
            U_236[:, 0:r] = U_233
            U_236[:, r:2*r] = U_233
            
            for j in range(m2):
                for k in range(p2):
                    V_236[j*6 + k, 0:r] = V_233[j, 0:r]
                    V_236[j*6 + (k+3), r:2*r] = V_233[j, 0:r]
                    
            for k in range(p2):
                for i in range(n2):
                    W_236[k*2 + i, 0:r] = W_233[k, 0:r]
                    W_236[(k+3)*2 + i, r:2*r] = W_233[k, 0:r]
                    
            return {"U": U_236.tolist(), "V": V_236.tolist(), "W": W_236.tolist()}
            
    return {"U": best_U, "V": best_V, "W": best_W}
