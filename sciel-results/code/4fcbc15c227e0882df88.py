import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Pre-compute consistent triples
    triples = []
    consistent_flags = []
    for x in range(nm):
        for y in range(mp):
            for z in range(pn):
                i_x, j_x = divmod(x, m)
                j_y, k_y = divmod(y, p)
                k_z, i_z = divmod(z, n)
                if j_x == j_y and k_y == k_z and i_z == i_x:
                    triples.append((x, y, z))
                    consistent_flags.append(1)
                else:
                    triples.append((x, y, z))
                    consistent_flags.append(0)
    
    triples = np.array(triples, dtype=np.int32)
    flags = np.array(consistent_flags, dtype=np.float64)
    num_triples = len(triples)
    
    def compute_residual(U_mat, V_mat, W_mat):
        """Compute residual for given matrices"""
        M = U_mat[:, :, None] * V_mat[None, :, :]
        Res = np.einsum('xyr,zr->xyp', M, W_mat)
        Res_flat = Res[triples[:, 0], triples[:, 1], triples[:, 2]]
        return Res_flat - flags
    
    def verify_solution(U_sol, V_sol, W_sol, tol=1e-9):
        """Verify if solution is exact with half-integer rounding"""
        U_round = np.round(U_sol * 2) / 2
        V_round = np.round(V_sol * 2) / 2
        W_round = np.round(W_sol * 2) / 2
        
        M_round = U_round[:, :, None] * V_round[None, :, :]
        Res_round = np.einsum('xyr,zr->xyp', M_round, W_round)
        Res_flat_round = Res_round[triples[:, 0], triples[:, 1], triples[:, 2]]
        
        if np.allclose(Res_flat_round, flags, atol=tol):
            return U_round, V_round, W_round, True
        return None, None, None, False
    
    def alternating_minimization(U, V, W, max_iter=100, tol=1e-6):
        """Alternating minimization to polish solution"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Fix V and W, solve for U
            # U[x,r] = sum_{y,z} V[y,r] W[z,r] target[x,y,z] / sum_{y,z} V[y,r]^2 W[z,r]^2
            # This is a linear system for each r
            U_new = np.zeros_like(U)
            for r in range(U.shape[1]):
                # Build linear system: sum_y V[y,r] * (sum_z W[z,r] * T[x,y,z]) = sum_y V[y,r] * (sum_z W[z,r] * target[x,y,z])
                # Simplified: use least squares
                A = np.zeros((nm, mp * pn))
                for idx, (x, y, z) in enumerate(triples):
                    A[x, y * pn + z] = V[y, r] * W[z, r]
                b = np.zeros(nm)
                for idx, (x, y, z) in enumerate(triples):
                    b[x] += V[y, r] * W[z, r] * flags[idx]
                
                # Solve A @ u = b via least squares
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[:, r] = u_sol
                except:
                    U_new[:, r] = U[:, r]
            U = U_new
            
            # Fix U and W, solve for V
            V_new = np.zeros_like(V)
            for r in range(V.shape[1]):
                A = np.zeros((mp, nm * pn))
                for idx, (x, y, z) in enumerate(triples):
                    A[y, x * pn + z] = U[x, r] * W[z, r]
                b = np.zeros(mp)
                for idx, (x, y, z) in enumerate(triples):
                    b[y] += U[x, r] * W[z, r] * flags[idx]
                
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[:, r] = v_sol
                except:
                    V_new[:, r] = V[:, r]
            V = V_new
            
            # Fix U and V, solve for W
            W_new = np.zeros_like(W)
            for r in range(W.shape[1]):
                A = np.zeros((pn, nm * mp))
                for idx, (x, y, z) in enumerate(triples):
                    A[z, x * mp + y] = U[x, r] * V[y, r]
                b = np.zeros(pn)
                for idx, (x, y, z) in enumerate(triples):
                    b[z] += U[x, r] * V[y, r] * flags[idx]
                
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[:, r] = w_sol
                except:
                    W_new[:, r] = W[:, r]
            W = W_new
            
            # Check convergence
            residual = compute_residual(U, V, W)
            cost = np.sum(residual ** 2)
            if cost < tol:
                break
        
        return U, V, W, cost
    
    # Try ranks from 41 down to 35
    for R in range(41, 34, -1):
        if time.time() > deadline - 10:
            break
        
        start_time = time.time()
        
        # Try multiple initializations
        best_cost = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Initialization 1: Schoolbook (if R >= 42) or partial
        if R >= 42:
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < R:
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
                        else:
                            break
                    if col >= R: break
                if col >= R: break
            
            residual = compute_residual(U, V, W)
            cost = np.sum(residual ** 2)
            if cost < best_cost:
                best_cost = cost
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Initializations 2-5: Random with different scales
        for attempt in range(4):
            if time.time() > deadline - 15:
                break
            
            scale = 0.1 * (attempt + 1)
            U = np.random.randn(nm, R) * scale
            V = np.random.randn(mp, R) * scale
            W = np.random.randn(pn, R) * scale
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Try alternating minimization first
            U_am, V_am, W_am, cost_am = alternating_minimization(U, V, W, max_iter=50)
            if cost_am < best_cost:
                best_cost = cost_am
                best_U, best_V, best_W = U_am, V_am, W_am
            
            # If good enough, try least_squares polish
            if cost_am < 1.0:
                x0 = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])
                
                def residual(params):
                    U_flat = params[:nm * R]
                    V_flat = params[nm * R : nm * R + mp * R]
                    W_flat = params[nm * R + mp * R :]
                    
                    U_mat = U_flat.reshape(nm, R)
                    V_mat = V_flat.reshape(mp, R)
                    W_mat = W_flat.reshape(pn, R)
                    
                    return compute_residual(U_mat, V_mat, W_mat)
                
                try:
                    result = least_squares(residual, x0, method='lm', max_nfev=5000)
                    if result.cost < best_cost:
                        U_sol = result.x[:nm * R].reshape(nm, R)
                        V_sol = result.x[nm * R : nm * R + mp * R].reshape(mp, R)
                        W_sol = result.x[nm * R + mp * R :].reshape(pn, R)
                        
                        U_polished, V_polished, W_polished, cost_polished = alternating_minimization(
                            U_sol, V_sol, W_sol, max_iter=100)
                        
                        if cost_polished < best_cost:
                            best_cost = cost_polished
                            best_U, best_V, best_W = U_polished, V_polished, W_polished
                except Exception as e:
                    pass
        
        # Check if best solution is valid
        if best_U is not None and best_cost < 1e-6:
            U_round, V_round, W_round, valid = verify_solution(best_U, best_V, best_W)
            if valid:
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        
        # If time is running out, break
        if time.time() - start_time > 20:
            continue
    
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}
