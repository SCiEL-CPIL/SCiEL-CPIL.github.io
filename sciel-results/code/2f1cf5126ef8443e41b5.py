import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Precompute index mappings
    # U: (n*m, r), index i*m + j
    # V: (m*p, r), index j*p + k
    # W: (p*n, r), index k*n + i
    
    def verify(U, V, W):
        # Exact verification using integer arithmetic (entries are half-integers)
        # Multiply by 2 to get integers
        U2 = np.round(np.array(U) * 2)
        V2 = np.round(np.array(V) * 2)
        W2 = np.round(np.array(W) * 2)
        # Check they're integers
        if not (np.allclose(U2, np.round(U2)) and np.allclose(V2, np.round(V2)) and np.allclose(W2, np.round(W2))):
            return False
        U2 = U2.astype(int)
        V2 = V2.astype(int)
        W2 = W2.astype(int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for alpha in range(U2.shape[1]):
                        s += U2[i*m+j, alpha] * V2[j*p+k, alpha] * W2[k*n+i, alpha]
                    # The product should be 1, but with the 2x scaling, it's 8 (since each factor is doubled)
                    # Actually, U = U2/2, V = V2/2, W = W2/2, so U*V*W = U2*V2*W2/8
                    # We need sum U*V*W = 1, so sum U2*V2*W2 = 8
                    if s != 8:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    best = None
    best_rank = n * m * p  # schoolbook
    U_s, V_s, W_s, best_rank = schoolbook()
    best = (U_s, V_s, W_s, best_rank)
    
    rng = np.random.default_rng(42)
    
    # Try ranks from low to high
    # Known rank for 3x3x4 is 20
    ranks_to_try = [20, 21, 22, 23, 24]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # More restarts for lower ranks
        max_seeds = 100 if rank <= 21 else 50
        
        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            # Scale matters. 1.0 is a good start.
            U = rng.standard_normal((n*m, rank))
            V = rng.standard_normal((m*p, rank))
            W = rng.standard_normal((p*n, rank))
            
            # Vectorized ALS
            # We solve for W, then U, then V in each iteration
            converged = False
            for it in range(300):
                # 1. Solve for W
                # For each (i, k), we have a system A W_{k,i} = 1
                # A is (m, rank) where A[j, alpha] = U[i*m+j, alpha] * V[j*p+k, alpha]
                for i in range(n):
                    for k in range(p):
                        # Construct A: (m, rank)
                        # U rows: i*m + j for j in 0..m-1
                        # V rows: j*p + k for j in 0..m-1
                        # We can vectorize this:
                        # U_i = U[i*m : i*m+m, :]
                        # V_k = V[:, k::p]  # This is not quite right, V index is j*p+k
                        # Let's just loop j to construct A, it's small (m=3)
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        # Solve A w = ones(m)
                        w, *_ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n + i, :] = w
                
                # 2. Solve for U
                # For each (i, j), system B U_{i,j} = 1
                # B is (p, rank) where B[k, alpha] = V[j*p+k, alpha] * W[k*n+i, alpha]
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((p, rank))
                        for k in range(p):
                            B[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        u, *_ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U[i*m + j, :] = u
                        
                # 3. Solve for V
                # For each (j, k), system C V_{j,k} = 1
                # C is (n, rank) where C[i, alpha] = U[i*m+j, alpha] * W[k*n+i, alpha]
                for j in range(m):
                    for k in range(p):
                        C = np.zeros((n, rank))
                        for i in range(n):
                            C[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        v, *_ = np.linalg.lstsq(C, np.ones(n), rcond=None)
                        V[j*p + k, :] = v
                
                # Check residual
                # We can compute max deviation
                max_dev = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for alpha in range(rank):
                                s += U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha]
                            dev = abs(s - 1.0)
                            if dev > max_dev:
                                max_dev = dev
                if max_dev < 1e-6:
                    converged = True
                    break
            
            if converged:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    best = (U_r, V_r, W_r, rank)
                    best_rank = rank
                    # If we found the minimum expected rank, we can stop
                    if rank == 20:
                        break
        
        if best_rank == 20:
            break

    U, V, W, r = best
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
