import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Target tensor T[i, j, j_prime, k] = delta_{j, j_prime}
    # Shape: (n, m, m, p)
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k] = 1.0
                
    def compute_M(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        M = np.zeros((n, m, m, p))
        for r in range(R):
            # Reshape to (n, m), (m, p), (p, n)
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            
            # Compute contribution: sum_{i,j,j',k} w_r[k,i] u_r[i,j] v_r[j',k]
            # Using einsum: 'ki,ij,ak->ijak'
            # w_r is (p, n) -> indices k, i
            # u_r is (n, m) -> indices i, j
            # v_r is (m, p) -> indices a, k (where a is j')
            term = np.einsum('ki,ij,ak->ijak', w_r, u_r, v_r)
            M += term
        return M
        
    def residuals(params, R):
        # params: flattened U, V, W
        # U: nm*R, V: mp*R, W: pn*R
        size_U = nm * R
        size_V = mp * R
        U = params[:size_U].reshape(nm, R)
        V = params[size_U:size_U + size_V].reshape(mp, R)
        W = params[size_U + size_V:].reshape(pn, R)
        
        M = compute_M(U, V, W, R)
        return (M - T).flatten()
        
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find lower rank
    # We will try R from 23 down to 10
    # But be careful with time.
    
    for R in range(rank_sb - 1, 10, -1):
        if time.time() > deadline:
            break
            
        # Initialize with schoolbook truncated
        # If R < rank_sb, we take the first R columns
        # This is a valid starting point, though not optimal
        U0 = U_sb[:, :R]
        V0 = V_sb[:, :R]
        W0 = W_sb[:, :R]
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residuals, params0, args=(R,), method='lm', max_nfev=1000)
            
            if result.success or result.cost < 1e-8:
                # Check if it's actually valid
                U_opt = result.x[:nm * R].reshape(nm, R)
                V_opt = result.x[nm * R: nm * R + mp * R].reshape(mp, R)
                W_opt = result.x[nm * R + mp * R:].reshape(pn, R)
                
                M_opt = compute_M(U_opt, V_opt, W_opt, R)
                err = np.linalg.norm(M_opt - T)
                
                if err < 1e-6:
                    # Found a valid decomposition of rank R
                    # Round to nearest half-integer?
                    # The problem asks for integer/half-integer entries.
                    # The optimizer finds floats. We should check if they are close to half-integers.
                    # If not, we might have a valid float decomposition, but the task requires integer/half-integer.
                    # However, if the true solution is rational with small denominator, it should be close.
                    
                    # Let's check if entries are close to 0.5 * integer
                    U_close = np.allclose(U_opt * 2, np.round(U_opt * 2), atol=1e-5)
                    V_close = np.allclose(V_opt * 2, np.round(V_opt * 2), atol=1e-5)
                    W_close = np.allclose(W_opt * 2, np.round(W_opt * 2), atol=1e-5)
                    
                    if U_close and V_close and W_close:
                        U_int = np.round(U_opt * 2) / 2.0
                        V_int = np.round(V_opt * 2) / 2.0
                        W_int = np.round(W_opt * 2) / 2.0
                        
                        # Verify exactness with rounded values
                        M_check = compute_M(U_int, V_int, W_int, R)
                        if np.allclose(M_check, T, atol=1e-9):
                            best_rank = R
                            best_U = U_int
                            best_V = V_int
                            best_W = W_int
                            # We can break if we found a very low rank, or continue to find even lower
                            # Given time constraints, we might stop early if R is small
                            if R <= 15:
                                break
                    else:
                        # Found a float solution, but not half-integer.
                        # We could try to snap to half-integers and check
                        U_snap = np.round(U_opt * 2) / 2.0
                        V_snap = np.round(V_opt * 2) / 2.0
                        W_snap = np.round(W_opt * 2) / 2.0
                        
                        M_snap = compute_M(U_snap, V_snap, W_snap, R)
                        if np.allclose(M_snap, T, atol=1e-9):
                            best_rank = R
                            best_U = U_snap
                            best_V = V_snap
                            best_W = W_snap
                            if R <= 15:
                                break
                            
        except Exception:
            pass

    # Convert best solution to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
