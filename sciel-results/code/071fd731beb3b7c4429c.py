import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
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
    
    triples = np.array(triples)
    flags = np.array(consistent_flags)
    num_triples = len(triples)
    
    # We know schoolbook is 42. Try to find lower ranks.
    # Literature suggests R(2,3,7) <= 38.
    # Let's try ranks from 38 down to 35.
    
    # Deadline guard
    deadline = time.time() + 140.0 # Leave 10s margin
    
    # Helper to compute residual
    def compute_residual(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        M = U[:, :, None] * V[None, :, :]
        Res = np.einsum('xyr,zr->xyp', M, W)
        Res_flat = Res[triples[:, 0], triples[:, 1], triples[:, 2]]
        return Res_flat - flags

    # Helper to verify and round
    def verify_and_round(U, V, W, R):
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        M_round = U_round[:, :, None] * V_round[None, :, :]
        Res_round = np.einsum('xyr,zr->xyp', M_round, W_round)
        Res_flat_round = Res_round[triples[:, 0], triples[:, 1], triples[:, 2]]
        
        if np.allclose(Res_flat_round, flags, atol=1e-9):
            return U_round, V_round, W_round
        return None

    # Try ranks from 38 down to 35
    for R in range(38, 34, -1):
        if time.time() > deadline - 15: # Need at least 15s left for finalization/verification
            break
            
        # Try multiple random restarts
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W
            # Scale random noise to be comparable to schoolbook entries (which are 0 or 1)
            # Standard deviation of 0.5 is a reasonable guess
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Flatten for optimizer
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residual(params):
                U_flat = params[:nm * R]
                V_flat = params[nm * R : nm * R + mp * R]
                W_flat = params[nm * R + mp * R :]
                
                U_mat = U_flat.reshape(nm, R)
                V_mat = V_flat.reshape(mp, R)
                W_mat = W_flat.reshape(pn, R)
                
                return compute_residual(U_mat, V_mat, W_mat, R)

            # Optimize
            try:
                # Use 'lm' for Levenberg-Marquardt
                result = least_squares(residual, x0, method='lm', max_nfev=5000)
                
                if result.cost < 1e-6: # Threshold for considering it a solution
                    U_sol = result.x[:nm * R].reshape(nm, R)
                    V_sol = result.x[nm * R : nm * R + mp * R].reshape(mp, R)
                    W_sol = result.x[nm * R + mp * R :].reshape(pn, R)
                    
                    rounded = verify_and_round(U_sol, V_sol, W_sol, R)
                    if rounded is not None:
                        U_round, V_round, W_round = rounded
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        
            except Exception as e:
                pass

    # Fallback to schoolbook (Rank 42)
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
