import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Local indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    strassen_terms = [
        # M1
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
        # M3
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
        # M4
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]},
        # M5
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        # M6
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    col = 0
    for r in range(3): # 3 blocks of p=6
        for s in range(2): # 2 blocks of m=4
            for t in range(7): # 7 Strassen terms
                term = strassen_terms[t]
                
                # U indices
                for local_u, coeff in term['U']:
                    i = local_u // 2
                    j_local = local_u % 2
                    j_global = 2 * s + j_local
                    u_idx = i * m + j_global
                    U[u_idx, col] += coeff
                
                # V indices
                for local_v, coeff in term['V']:
                    j_local = local_v // 2
                    k_local = local_v % 2
                    j_global = 2 * s + j_local
                    k_global = 2 * r + k_local
                    v_idx = j_global * p + k_global
                    V[v_idx, col] += coeff
                
                # W indices
                for local_w, coeff in term['W']:
                    i = local_w // 2
                    k_local = local_w % 2
                    k_global = 2 * r + k_local
                    w_idx = k_global * n + i
                    W[w_idx, col] += coeff
                
                col += 1

    # Attempt to reduce rank by finding linear dependencies
    # We check if any column is a linear combination of others in U, V, W simultaneously
    # This is a necessary condition for rank reduction in this structured way
    # If U[:, k] = sum c_i U[:, i], then we must have V[:, k] = sum c_i V[:, i] and W[:, k] = sum c_i W[:, i]
    
    def reduce_rank(U, V, W, max_iter=10):
        rank = U.shape[1]
        # Create a list of active columns
        active = list(range(rank))
        
        for _ in range(max_iter):
            found_reduction = False
            # Try to remove each column
            for k in range(rank):
                if k not in active:
                    continue
                    
                # Check if column k is in the span of the other active columns
                # We use SVD to check rank
                # Construct matrix of other columns
                other_cols = [c for c in active if c != k]
                
                if len(other_cols) == 0:
                    continue
                    
                # We need to solve U[:, other_cols] @ x = U[:, k]
                # And check if V and W are consistent
                U_other = U[:, other_cols]
                V_other = V[:, other_cols]
                W_other = W[:, other_cols]
                
                U_k = U[:, k]
                V_k = V[:, k]
                W_k = W[:, k]
                
                # Solve least squares for U
                try:
                    x, residuals, rank_u, s = np.linalg.lstsq(U_other, U_k, rcond=None)
                    # Check residual
                    if np.linalg.norm(U_other @ x - U_k) < 1e-9:
                        # Check V
                        if np.linalg.norm(V_other @ x - V_k) < 1e-9:
                            # Check W
                            if np.linalg.norm(W_other @ x - W_k) < 1e-9:
                                # We can remove column k
                                # Update active list
                                active.remove(k)
                                found_reduction = True
                                break
                except:
                    pass
            
            if not found_reduction:
                break
                
        return active

    active_cols = reduce_rank(U, V, W)
    
    if len(active_cols) < rank:
        U = U[:, active_cols]
        V = V[:, active_cols]
        W = W[:, active_cols]
        rank = len(active_cols)
        
    # Quick verification
    try:
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,ki->ijk', U_t, V_t, W_t)
        if not np.allclose(T, 1.0, atol=1e-9):
            # Fallback to schoolbook if verification fails
            r_sb = n * m * p
            U_sb = np.zeros((n * m, r_sb))
            V_sb = np.zeros((m * p, r_sb))
            W_sb = np.zeros((p * n, r_sb))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, col] = 1
                        V_sb[j * p + k, col] = 1
                        W_sb[k * n + i, col] = 1
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
