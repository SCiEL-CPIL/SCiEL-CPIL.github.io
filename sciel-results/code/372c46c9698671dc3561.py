import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] where C[i, k] = sum_j A[i, j] B[j, k]
    # T[i, j, k] = 1 if the term A[i, j] * B[j, k] contributes to C[i, k]
    # In the decomposition C = sum_r U[:, r] (A * V[:, r]) W[:, r]? 
    # No, the standard decomposition is:
    # C_{ik} = sum_{r} W_{ik, r} * (sum_j U_{ij, r} A_{ij}) * (sum_k V_{jk, r} B_{jk})?
    # No, the factor matrices are U (n*m, R), V (m*p, R), W (p*n, R).
    # The equation is:
    # sum_{r=0}^{R-1} W[:, r] * ( U[:, r]^T A_flat ) * ( V[:, r]^T B_flat ) ?
    # No.
    # The tensor T is decomposed as T = sum_r u_r \otimes v_r \otimes w_r.
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Note the index mapping for W: W[k*n+i, r].
    
    # Let's verify the index mapping from the prompt:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # U shape (n*m, rank) -> index i*m+j
    # V shape (m*p, rank) -> index j*p+k
    # W shape (p*n, rank) -> index k*n+i
    
    # So T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Target T[i, j, k] is 1 for all i, j, k?
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The term A[i, j] B[j, k] appears in C[i, k].
    # So yes, T[i, j, k] = 1.
    
    # Construct Target Tensor
    T = np.ones((n, m, p))
    
    # Schoolbook Decomposition (Rank 36)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
                
    # Function to verify a decomposition
    def verify(U, V, W):
        # U, V, W are lists of lists or numpy arrays
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check if entries are integers or half-integers
        # Multiply by 2, check if close to integer
        if not np.allclose(U_np * 2, np.round(U_np * 2)): return False
        if not np.allclose(V_np * 2, np.round(V_np * 2)): return False
        if not np.allclose(W_np * 2, np.round(W_np * 2)): return False
        
        # Compute Tensor Sum
        # T_calc[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can do this with einsum
        # U is (nm, R), V is (mp, R), W is (pn, R)
        # We want to sum over R.
        # Let's reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U_np.reshape(n, m, -1)
        V_reshaped = V_np.reshape(m, p, -1)
        W_reshaped = W_np.reshape(p, n, -1)
        
        # T_calc[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
        # Note: W index is k*n+i. In W_reshaped, the second index is n (i), first is p (k).
        # So W_reshaped[k, i, r] is correct.
        
        # Use einsum: 'ijr,jkr,kir->ijk'
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if T_calc equals T (which is all 1s)
        return np.allclose(T_calc, T, atol=1e-9)

    # Check schoolbook
    if verify(U_sb, V_sb, W_sb):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_schoolbook
    else:
        # Should not happen
        best_U, best_V, best_W = None, None, None
        best_rank = -1

    # Attempt to find lower rank
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Try ranks from 35 down to 20
    for rank in range(rank_schoolbook - 1, 19, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Optimization
        # Variables: U (nm*R), V (mp*R), W (pn*R)
        # Total vars: R * (nm + mp + pn)
        
        # Initial guess: Random or scaled schoolbook?
        # Random is safer to avoid local minima of "zero" solution
        # But we want to find a solution.
        # Let's try a few random restarts for each rank.
        
        found = False
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            np.random.seed(42 + rank + restart)
            U_init = np.random.randn(n*m, rank) * 0.1
            V_init = np.random.randn(m*p, rank) * 0.1
            W_init = np.random.randn(p*n, rank) * 0.1
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Residual function
            def residual(x):
                U_flat, V_flat, W_flat = np.split(x, [n*m*rank, (n*m + m*p)*rank])
                U_r = U_flat.reshape(n, m, rank)
                V_r = V_flat.reshape(m, p, rank)
                W_r = W_flat.reshape(p, n, rank)
                
                T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                return (T_calc - T).flatten()
                
            try:
                from scipy.optimize import least_squares
                result = least_squares(residual, x0, method='lm', max_nfev=1000)
                
                if result.success or result.cost < 1e-6:
                    # Extract solution
                    U_sol, V_sol, W_sol = np.split(result.x, [n*m*rank, (n*m + m*p)*rank])
                    U_sol = U_sol.reshape(n*m, rank)
                    V_sol = V_sol.reshape(m*p, rank)
                    W_sol = W_sol.reshape(p*n, rank)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify snapped solution
                    if verify(U_snap, V_snap, W_snap):
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = rank
                        found = True
                        break # Found for this rank, try lower?
                        # The task asks for rank-minimal. So we should continue to lower ranks.
                        # But lower ranks are harder to find.
                        # Let's break out of restart loop and continue rank loop.
            except Exception:
                pass
        
        if found:
            # Continue to lower rank
            continue
        else:
            # If we couldn't find rank 'rank', it's unlikely we'll find rank-1 easily without more time
            # But let's try one lower rank if time permits
            if time.time() - start_time > time_limit - 20:
                break

    # Return best found
    if best_rank > 0:
        # Convert to lists of lists as required?
        # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # The seed returns lists of lists.
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
