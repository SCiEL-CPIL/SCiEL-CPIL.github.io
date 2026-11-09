import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24
    
    # Build sparse tensor indices
    idx_i, idx_j, idx_k = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_i.append(i*m+j)
                idx_j.append(j*p+k)
                idx_k.append(k*n+i)
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)
    num_eq = len(idx_i)  # 120
    
    # Schoolbook solution
    rank_school = num_eq
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    for c in range(rank_school):
        U[idx_i[c], c] = 1.0
        V[idx_j[c], c] = 1.0
        W[idx_k[c], c] = 1.0
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try to find lower rank using alternating minimization
    # For efficiency, only evaluate on the 120 non-zero entries
    def compute_error(U, V, W):
        # For each (i,j,k) in the support, compute sum_c U[i,c]*V[j,c]*W[k,c]
        # Vectorized: for each c, compute U[idx_i,c] * V[idx_j,c] * W[idx_k,c] and sum
        err = np.zeros(num_eq)
        for c in range(U.shape[1]):
            err += U[idx_i, c] * V[idx_j, c] * W[idx_k, c]
        return np.max(np.abs(err - 1.0))
    
    np.random.seed(123)
    
    # Try ranks from 119 down to 100
    for target_rank in range(119, 95, -1):
        if time.time() > deadline:
            break
        
        success = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Alternating minimization
            for it in range(300):
                if time.time() > deadline:
                    break
                
                # Update U: for each row a, solve min_Ua sum_{(i,j,k):i=a} (sum_c Ua_c V_j_c W_k_c - 1)^2
                # This is a quadratic in Ua
                # Let's do it row by row for efficiency
                for a in range(nm):
                    # Find all (j,k) such that there exists i with i*m+j... wait, a = i*m+j
                    # So for fixed a, j is determined: j = a % m, i = a // m
                    # Then k ranges over 0..p-1, and the constraint is on W[k*n+i]
                    j_a = a % m
                    i_a = a // m
                    # The equations involving U[a,:]: for each k, sum_c U[a,c] * V[j_a*p+k, c] * W[k*n+i_a, c] = 1
                    # Let Vrow = V[j_a*p + np.arange(p), :]  # (p, target_rank)
                    # Let Wrows = W[np.arange(p)*n + i_a, :]  # (p, target_rank)
                    # For each k: sum_c U[a,c] * Vrow[k,c] * Wrows[k,c] = 1
                    # Let M = Vrow * Wrows  # (p, target_rank), element-wise
                    # Then: U[a,:] @ M.T = ones(p)
                    Vrow = V[j_a*p:(j_a+1)*p, :]  # (p, target_rank)
                    Wrows = W[np.arange(p)*n + i_a, :]  # (p, target_rank)
                    M = Vrow * Wrows  # (p, target_rank)
                    rhs = np.ones(p)
                    try:
                        U[a,:] = np.linalg.lstsq(M.T, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update V similarly
                for b in range(mp):
                    # b = j*p + k, so j = b // p, k = b % p
                    j_b = b // p
                    k_b = b % p
                    # Equations: for each i, sum_c U[i*m+j_b, c] * V[b,c] * W[k_b*n+i, c] = 1
                    Urows = U[np.arange(n)*m + j_b, :]  # (n, target_rank)
                    Wrow = W[k_b*n:(k_b+1)*n, :]  # (n, target_rank)
                    M = Urows * Wrow  # (n, target_rank)
                    rhs = np.ones(n)
                    try:
                        V[b,:] = np.linalg.lstsq(M.T, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update W similarly
                for c_idx in range(pn):
                    # c_idx = k*n + i, so k = c_idx // n, i = c_idx % n
                    k_c = c_idx // n
                    i_c = c_idx % n
                    # Equations: for each j, sum_c U[i_c*m+j, c] * V[j*p+k_c, c] * W[c_idx,c] = 1
                    Urows = U[i_c*m:(i_c+1)*m, :]  # (m, target_rank)
                    Vrows = V[np.arange(m)*p + k_c, :]  # (m, target_rank)
                    M = Urows * Vrows  # (m, target_rank)
                    rhs = np.ones(m)
                    try:
                        W[c_idx,:] = np.linalg.lstsq(M.T, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Check error
                if it % 10 == 0:
                    err = compute_error(U, V, W)
                    if err < 1e-10:
                        # Round to half-integers
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        err_r = compute_error(U_r, V_r, W_r)
                        if err_r < 1e-6:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                            success = True
                            break
                        break
            if success:
                break
        
        if success:
            continue
        else:
            # If we couldn't find at this rank, try a couple more lower ranks
            if target_rank < 110:
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
