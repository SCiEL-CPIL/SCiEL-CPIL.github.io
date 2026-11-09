"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Helper to verify exact decomposition (within floating point tolerance)
    def is_valid(U, V, W, rank):
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Exploit the structure C = A*B where B is 4x8.
    # Split B into two 4x4 blocks: B = [B1, B2].
    # C = A*B1 + A*B2.
    # If we have a rank R decomposition for 4x4*4x4 -> 4x4, we can construct a rank 2R decomposition for 4x4*4x8 -> 4x8.
    # U_448 = [U_444, U_444]
    # V_448 = [V_444_1, V_444_2] where V_444_1 maps to B1 indices and V_444_2 to B2 indices.
    # W_448 = [W_444, W_444]
    
    # We need to find a low-rank decomposition for 4x4*4x4.
    # Schoolbook rank is 64.
    # Known lower bound is ~49.
    # Let's try to find a 4x4*4x4 decomposition with rank R44.
    # Then the total rank is 2*R44.
    # If we find R44=49, total rank is 98.
    # If we find R44=50, total rank is 100.
    # Schoolbook for 4x4x8 is 128.
    # So even R44=55 gives rank 110, which is better than 128.
    
    # Let's try to optimize the 4x4*4x4 problem first.
    n4, m4, p4 = 4, 4, 4
    
    def find_444_decomposition(target_rank, max_time=10):
        start_time = time.time()
        U_init = np.random.randn(n4 * m4, target_rank) * 0.1
        V_init = np.random.randn(m4 * p4, target_rank) * 0.1
        W_init = np.random.randn(p4 * n4, target_rank) * 0.1
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals_444(params):
            U_flat = params[:n4*m4*target_rank]
            V_flat = params[n4*m4*target_rank : n4*m4*target_rank + m4*p4*target_rank]
            W_flat = params[n4*m4*target_rank + m4*p4*target_rank :]
            
            U_mat = U_flat.reshape(n4*m4, target_rank)
            V_mat = V_flat.reshape(m4*p4, target_rank)
            W_mat = W_flat.reshape(p4*n4, target_rank)
            
            res = np.zeros(n4*m4*p4)
            idx = 0
            for i in range(n4):
                for j in range(m4):
                    for k in range(p4):
                        u_row = U_mat[i*m4+j, :]
                        v_row = V_mat[j*p4+k, :]
                        w_row = W_mat[k*n4+i, :]
                        val = np.dot(u_row, v_row * w_row)
                        res[idx] = val - 1.0
                        idx += 1
            return res
        
        try:
            result = least_squares(residuals_444, x0, method='lm', max_nfev=500, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            
            if result.cost < 1e-8:
                U_sol = result.x[:n4*m4*target_rank].reshape(n4*m4, target_rank)
                V_sol = result.x[n4*m4*target_rank : n4*m4*target_rank + m4*p4*target_rank].reshape(m4*p4, target_rank)
                W_sol = result.x[n4*m4*target_rank + m4*p4*target_rank :].reshape(p4*n4, target_rank)
                
                # Verify 4x4x4
                valid = True
                for i in range(n4):
                    for j in range(m4):
                        for k in range(p4):
                            val = np.dot(U_sol[i*m4+j, :], V_sol[j*p4+k, :] * W_sol[k*n4+i, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        return None

    # Try to find a 4x4x4 decomposition with rank R44
    # We want 2*R44 < 128 => R44 < 64.
    # Let's try R44 from 50 to 63.
    found_444 = None
    best_R44 = 64 # Schoolbook for 4x4x4
    
    for R44 in range(50, 64):
        if time.time() > deadline - 20:
            break
        if R44 >= best_R44:
            continue
            
        # Try a few restarts
        for restart in range(2):
            if time.time() > deadline - 10:
                break
            np.random.seed(42 + R44 * 10 + restart)
            res = find_444_decomposition(R44)
            if res is not None:
                found_444 = res
                best_R44 = R44
                break
        if found_444 is not None:
            break

    if found_444 is not None:
        U44, V44, W44 = found_444
        R44 = best_R44
        R448 = 2 * R44
        
        # Construct 4x4x8 decomposition
        U_448 = np.zeros((n * m, R448))
        V_448 = np.zeros((m * p, R448))
        W_448 = np.zeros((p * n, R448))
        
        U_448[:, :R44] = U44
        U_448[:, R44:] = U44
        
        # V construction:
        # V_448[j*8+k, t] for k in 0..3 should be V44[j*4+k, t-R44] for t in R44..2R44-1?
        # Wait, the decomposition is C = A*B1 + A*B2.
        # Term 1: A*B1. U is U44. V is V44 mapping B1. W is W44.
        # Term 2: A*B2. U is U44. V is V44 mapping B2. W is W44.
        
        # So for t in 0..R44-1:
        # U_448[i*4+j, t] = U44[i*4+j, t]
        # V_448[j*8+k, t] = V44[j*4+k, t] if k<4 (B1)
        # W_448[k*4+i, t] = W44[k*4+i, t]
        
        # For t in R44..2R44-1:
        # U_448[i*4+j, t] = U44[i*4+j, t-R44]
        # V_448[j*8+k, t] = V44[j*4+(k-4), t-R44] if k>=4 (B2)
        # W_448[k*4+i, t] = W44[k*4+i, t-R44]
        
        for i in range(4):
            for j in range(4):
                for t in range(R44):
                    U_448[i*4+j, t] = U44[i*4+j, t]
                    U_448[i*4+j, R44+t] = U44[i*4+j, t]
        
        for j in range(4):
            for k in range(4):
                for t in range(R44):
                    V_448[j*8+k, t] = V44[j*4+k, t]
                    V_448[j*8+(k+4), R44+t] = V44[j*4+k, t]
        
        for k in range(4):
            for i in range(4):
                for t in range(R44):
                    W_448[k*4+i, t] = W44[k*4+i, t]
                    W_448[(k+4)*4+i, R44+t] = W44[k*4+i, t]
        
        # Verify 4x4x8
        valid = True
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    val = np.dot(U_448[i*4+j, :], V_448[j*8+k, :] * W_448[k*4+i, :])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            best_U, best_V, best_W = U_448, V_448, W_448
            best_rank = R448

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
