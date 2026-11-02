# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    n, m, p = 4, 4, 8
    
    # 1. Construct Schoolbook Decomposition (Rank 128)
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
                
    def to_list(mat):
        return mat.tolist()
        
    best_U = to_list(U_school)
    best_V = to_list(V_school)
    best_W = to_list(W_school)
    best_rank = rank_school
    
    # 2. Try to find lower rank via structured search
    # For M_{4,4,8}, try rank 44 (two independent M_{4,4,4} blocks)
    # and other candidate ranks
    
    def verify_decomposition(U, V, W, n, m, p):
        """Check if U⊗V⊗W equals the matmul tensor."""
        rank = len(W[0]) if isinstance(W, list) else W.shape[1]
        if isinstance(U, list):
            U = np.array(U)
        if isinstance(V, list):
            V = np.array(V)
        if isinstance(W, list):
            W = np.array(W)
            
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # Target: T[i,j,k] = 1 for all i,j,k
        error = np.max(np.abs(T_rec - 1.0))
        return error < 1e-10
    
    def try_rank(rank, max_restarts=3, max_iters=20):
        if rank >= best_rank:
            return None
        if time.time() > deadline - 5:
            return None
            
        for seed in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            rng = np.random.RandomState(seed * 7919 + rank)
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        V_slice = V[j*p : j*p+p, :]
                        W_slice = W[i::n, :]
                        A = V_slice * W_slice
                        b = np.ones(p)
                        try:
                            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U[i*m + j, :] = u
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        U_slice = U[j::m, :]
                        W_slice = W[k*n : k*n+n, :]
                        B = U_slice * W_slice
                        c = np.ones(n)
                        try:
                            v, _, _, _ = np.linalg.lstsq(B, c, rcond=None)
                            V[j*p + k, :] = v
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        U_slice = U[i*m : i*m+m, :]
                        V_slice = V[k::p, :]
                        C = U_slice * V_slice
                        d = np.ones(m)
                        try:
                            w, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
                            W[k*n + i, :] = w
                        except:
                            pass
                
                # Check error
                if verify_decomposition(U, V, W, n, m, p):
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
        return None
    
    # Try a few candidate ranks below 128
    # Known lower bounds: rank >= max(16, 32, 32) = 32
    # Try ranks that are multiples of small numbers
    candidate_ranks = [126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80]
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        res = try_rank(r)
        if res:
            best_U = res["U"]
            best_V = res["V"]
            best_W = res["W"]
            best_rank = r
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
