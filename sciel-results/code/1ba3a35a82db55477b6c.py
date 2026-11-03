import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Unfold T
    T1 = T.reshape(dim_u, dim_v * dim_w)
    T2 = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
    T3 = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_u, rank_schoolbook))
    V_school = np.zeros((dim_v, rank_schoolbook))
    W_school = np.zeros((dim_w, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 110 # 110s limit
    
    # Try ranks from 35 down to 25
    for R in range(rank_schoolbook - 1, 24, -1):
        if time.time() > deadline:
            break
            
        # Try restarts. Vectorized ALS is fast, so we can afford more.
        # 20 restarts should be feasible.
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Use uniform random in [-1, 1] or Gaussian?
            # Gaussian is fine.
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            # ALS iterations
            converged = False
            prev_err = float('inf')
            for it in range(300):
                if time.time() > deadline:
                    break
                    
                # Update U
                # M (dim_v * dim_w, R)
                # M[v*dim_w + w, r] = V[v, r] * W[w, r]
                M_3d = np.einsum('vr,wr->vwr', V, W)
                M = M_3d.reshape(dim_v * dim_w, R)
                
                try:
                    # U = argmin || T1 - U M^T ||_F
                    # U = T1 (M^T)^+
                    U = T1 @ np.linalg.pinv(M)
                except:
                    break
                    
                # Update V
                # N (dim_u * dim_w, R)
                # N[u*dim_w + w, r] = U[u, r] * W[w, r]
                N_3d = np.einsum('ur,wr->uwr', U, W)
                N = N_3d.reshape(dim_u * dim_w, R)
                
                try:
                    V = T2 @ np.linalg.pinv(N)
                except:
                    break
                    
                # Update W
                # P (dim_u * dim_v, R)
                # P[u*dim_v + v, r] = U[u, r] * V[v, r]
                P_3d = np.einsum('ur,vr->uvr', U, V)
                P = P_3d.reshape(dim_u * dim_v, R)
                
                try:
                    W = T3 @ np.linalg.pinv(P)
                except:
                    break
                    
                # Check convergence
                M_check = np.einsum('vr,wr->vwr', V, W).reshape(dim_v * dim_w, R)
                Res = T1 - U @ M_check.T
                err = np.linalg.norm(Res, 'fro')
                
                if abs(prev_err - err) < 1e-9 and err < 1e-8:
                    converged = True
                    break
                prev_err = err
                
            if converged:
                # Polish: Round to half-integers
                U_round = 2 * np.round(U / 2)
                V_round = 2 * np.round(V / 2)
                W_round = 2 * np.round(W / 2)
                
                # Verify exact decomposition with rounded values
                M_round = np.einsum('vr,wr->vwr', V_round, W_round).reshape(dim_v * dim_w, R)
                Res_round = T1 - U_round @ M_round.T
                if np.allclose(Res_round, 0, atol=1e-6):
                    # Check full tensor just in case
                    # T_recon = np.einsum('ur,vr,wr->uvw', U_round, V_round, W_round)
                    # if np.allclose(T_recon, T, atol=1e-6):
                    best_rank = R
                    best_result = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    break # Break restart loop to try lower R
                else:
                    # If rounding failed, this solution is invalid for the strict integer/half-integer requirement.
                    # We do not update best_result.
                    pass

    return best_result
