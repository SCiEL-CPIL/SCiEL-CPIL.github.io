"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import scipy.optimize

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin

    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = (U_school, V_school, W_school)
    best_rank = rank_school

    # Precompute target tensor T
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T_target[I, J, K] = 1.0

    # Helper to check if a candidate (U, V, W) is valid and snap it
    def try_snap_and_verify(U_opt, V_opt, W_opt, R):
        # Snap to half-integers
        U_snap = np.round(U_opt * 2) / 2.0
        V_snap = np.round(V_opt * 2) / 2.0
        W_snap = np.round(W_opt * 2) / 2.0
        
        # Verify exactness
        Res = np.einsum('ic,jc,kc->ijk', U_snap, V_snap, W_snap)
        err = np.linalg.norm(Res - T_target)
        if err < 1e-5:
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        return None

    # Search for lower ranks
    # We try ranks 74 down to 65
    # For each rank, we try a few random restarts with least_squares
    
    np.random.seed(42)
    
    # Ranks to try, ordered from low to high difficulty (or just descending rank)
    # Theoretical lower bound is around 50-60. 
    # Current best known for 3x5x5 might be around 60-70.
    # Let's scan 74 down to 60.
    
    for R in range(74, 59, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Number of restarts decreases as rank gets lower (harder) or time runs out
        # Start with 20 restarts for high ranks, fewer for low
        num_restarts = 15 if R > 70 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Scale: entries should be small
            # Uniform [-1, 1] is a good start
            U_init = np.random.uniform(-1, 1, (nm, R))
            V_init = np.random.uniform(-1, 1, (mp, R))
            W_init = np.random.uniform(-1, 1, (pn, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residual(x):
                U_flat = x[:nm*R]
                V_flat = x[nm*R:nm*R + mp*R]
                W_flat = x[nm*R + mp*R:]
                
                U = U_flat.reshape(nm, R)
                V = V_flat.reshape(mp, R)
                W = W_flat.reshape(pn, R)
                
                Res = np.einsum('ic,jc,kc->ijk', U, V, W)
                return (Res - T_target).flatten()
            
            try:
                # Use least_squares
                # Tighter tolerances for better snap chance
                res = scipy.optimize.least_squares(residual, x0, method='trf', 
                                                   ftol=1e-14, xtol=1e-14, gtol=1e-14,
                                                   max_nfev=500) # Limit iterations per restart to save time
                
                if res.success or res.cost < 1e-10:
                    U_opt = res.x[:nm*R].reshape(nm, R)
                    V_opt = res.x[nm*R:nm*R + mp*R].reshape(mp, R)
                    W_opt = res.x[nm*R + mp*R:].reshape(pn, R)
                    
                    snapped = try_snap_and_verify(U_opt, V_opt, W_opt, R)
                    if snapped:
                        U_list, V_list, W_list = snapped
                        # Final verification with _verify
                        if _verify(U_list, V_list, W_list, n, m, p):
                            best_result = (U_list, V_list, W_list)
                            best_rank = R
                            found = True
                            # If we found a very low rank, we can stop early?
                            # No, we want the MINIMAL rank. So continue to lower R.
                            break # Try next lower rank (R-1)
                            
            except Exception:
                pass

        if found:
            continue # Try lower rank

    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
