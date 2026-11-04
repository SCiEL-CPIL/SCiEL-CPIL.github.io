import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook solution
    rank_school = n*m*p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline
    
    # Try ranks from 30 to 74
    for rank in range(30, 75):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random initializations
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(trial * 10000 + rank * 100)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Alternating least squares
            converged = False
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                
                # Solve for V: for each b, M * V[b,:] = vec(T[:,b,:])
                V_new = np.zeros((mp, rank))
                for b in range(mp):
                    # M[:,r] = vec(U[:,r] ⊗ W[:,r])
                    M = np.zeros((nm*pn, rank))
                    for r in range(rank):
                        M[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                    rhs = T[:, b, :].flatten()
                    V_new[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                V = V_new
                
                # Solve for U: for each a, M * U[a,:] = vec(T[a,:,:])
                U_new = np.zeros((nm, rank))
                for a in range(nm):
                    M = np.zeros((mp*pn, rank))
                    for r in range(rank):
                        M[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                    rhs = T[a, :, :].flatten()
                    U_new[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                U = U_new
                
                # Solve for W: for each c, M * W[c,:] = vec(T[:,:,c])
                W_new = np.zeros((pn, rank))
                for c in range(pn):
                    M = np.zeros((nm*mp, rank))
                    for r in range(rank):
                        M[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                    rhs = T[:, :, c].flatten()
                    W_new[c, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                W = W_new
                
                # Check convergence
                T_rec = np.zeros((nm, mp, pn))
                for r in range(rank):
                    T_rec += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
                err = np.max(np.abs(T_rec - T))
                if err < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Try to round to half-integers
                for scale in [2, 1]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    
                    T_rec = np.zeros((nm, mp, pn))
                    for r in range(rank):
                        T_rec += np.outer(np.outer(U_r[:, r], V_r[:, r]), W_r[:, r])
                    err = np.max(np.abs(T_rec - T))
                    if err < 1e-9:
                        best_rank = rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        found = True
                        break
                if found:
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
