"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the matrix multiplication tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    schoolbook_rank = n * m * p
    best_U = np.zeros((nm, schoolbook_rank))
    best_V = np.zeros((mp, schoolbook_rank))
    best_W = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1.0
                best_V[j*p+k, col] = 1.0
                best_W[k*n+i, col] = 1.0
                col += 1
    best_rank = schoolbook_rank
    deadline = time.time() + 130
    
    def check_exact(U, V, W, r):
        T_rec = np.zeros((nm, mp, pn))
        for s in range(r):
            T_rec += np.einsum('i,j,k->ijk', U[:, s], V[:, s], W[:, s])
        return np.max(np.abs(T_rec - T))
    
    def try_rank(target_rank, n_attempts, deadline):
        for attempt in range(n_attempts):
            if time.time() > deadline - 15:
                return None
            np.random.seed(attempt * 7919 + target_rank * 104729)
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            for iteration in range(60):
                if time.time() > deadline - 15:
                    break
                # Update U: fix V, W
                M = np.einsum('bs,cs->sbc', V, W).reshape(target_rank, mp * pn)
                Y = T.reshape(nm, mp * pn)
                MMt = M @ M.T
                try:
                    U = Y @ M.T @ np.linalg.inv(MMt)
                except:
                    U = Y @ M.T @ np.linalg.pinv(MMt)
                
                # Update V: fix U, W
                M = np.einsum('as,cs->sac', U, W).reshape(target_rank, nm * pn)
                Y = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                MMt = M @ M.T
                try:
                    V = Y @ M.T @ np.linalg.inv(MMt)
                except:
                    V = Y @ M.T @ np.linalg.pinv(MMt)
                
                # Update W: fix U, V
                M = np.einsum('as,bs->sab', U, V).reshape(target_rank, nm * mp)
                Y = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                MMt = M @ M.T
                try:
                    W = Y @ M.T @ np.linalg.inv(MMt)
                except:
                    W = Y @ M.T @ np.linalg.pinv(MMt)
            
            err = check_exact(U, V, W, target_rank)
            if err < 1e-9:
                return (U, V, W, target_rank)
            if err < 1e-3:
                for scale in [2, 4, 10, 20]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    err_r = check_exact(U_r, V_r, W_r, target_rank)
                    if err_r < 1e-9:
                        return (U_r, V_r, W_r, target_rank)
        return None
    
    # Try to find lower rank decompositions
    for target_rank in range(schoolbook_rank - 1, max(20, schoolbook_rank // 2), -1):
        if time.time() > deadline - 20:
            break
        result = try_rank(target_rank, 15, deadline)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    U = best_U[:, :best_rank]
    V = best_V[:, :best_rank]
    W = best_W[:, :best_rank]
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
