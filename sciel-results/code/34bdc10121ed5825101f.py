import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook
    rank_sb = n*m*p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    def check(U, V, W, rank):
        for i in range(n):
            for k in range(p):
                for jp in range(m):
                    for jpp in range(m):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+jp, c] * V[jpp*p+k, c] * W[k*n+i, c]
                        expected = 1.0 if jp == jpp else 0.0
                        if abs(s - expected) > 1e-6:
                            return False
        return True
    
    half_ints = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for rank in range(24, best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        for trial in range(100):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(trial * 1000 + rank)
            U = np.random.choice(half_ints, size=(nm, rank))
            V = np.random.choice(half_ints, size=(mp, rank))
            
            # Solve for W
            # For each (i,k), we have: sum_c U[i*m+j', c] * V[j''*p+k, c] * W[k*n+i, c] = delta_{j',j''}
            # This is: sum_c (U[i*m+j', c] * V[j''*p+k, c]) * W[k*n+i, c] = delta_{j',j''}
            # Let M_{j',j'',c} = U[i*m+j', c] * V[j''*p+k, c]
            # Then: sum_c M_{j',j'',c} * W[k*n+i, c] = delta_{j',j''}
            # This is a system of m*m equations in r unknowns (W[k*n+i, c] for c=0..r-1)
            
            W = np.zeros((pn, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build the system
                    A_sys = np.zeros((m*m, rank))
                    b_sys = np.zeros(m*m)
                    for jp in range(m):
                        for jpp in range(m):
                            idx = jp * m + jpp
                            for c in range(rank):
                                A_sys[idx, c] = U[i*m+jp, c] * V[jpp*p+k, c]
                            b_sys[idx] = 1.0 if jp == jpp else 0.0
                    
                    # Solve A_sys * W_col = b_sys
                    if rank >= m*m:
                        # Overdetermined, use least squares
                        W_col, residuals, rank_sol, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-6:
                            valid = False
                            break
                    else:
                        # Underdetermined, use minimum norm solution
                        W_col, residuals, rank_sol, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-6:
                            valid = False
                            break
                    
                    W[k*n+i, :] = W_col
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Round W to half-integers
            W_r = np.round(W * 2) / 2
            
            if check(U, V, W_r, rank):
                best_rank = rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W_r.tolist()
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
