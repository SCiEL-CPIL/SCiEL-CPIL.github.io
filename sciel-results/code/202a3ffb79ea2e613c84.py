import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    deadline = start_time + 140
    
    # Search for rank 16 first, then 15, 14
    # Use a faster random search with structured sparsity
    for rank in [16, 15, 14]:
        if time.time() > deadline - 10:
            break
            
        max_trials = 20000
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            # Generate sparse random matrices
            # Density ~ 0.3
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # Randomly place non-zero values
            for c in range(rank):
                # U column
                nz_u = np.random.randint(1, 4)
                idx_u = np.random.choice(nm, nz_u, replace=False)
                U[idx_u, c] = np.random.choice([-1, -0.5, 0.5, 1], nz_u)
                
                # V column
                nz_v = np.random.randint(1, 4)
                idx_v = np.random.choice(mp, nz_v, replace=False)
                V[idx_v, c] = np.random.choice([-1, -0.5, 0.5, 1], nz_v)
                
                # W column
                nz_w = np.random.randint(1, 4)
                idx_w = np.random.choice(pn, nz_w, replace=False)
                W[idx_w, c] = np.random.choice([-1, -0.5, 0.5, 1], nz_w)
            
            # Compute R
            R = np.einsum('as,bs,cs->abc', U, V, W)
            
            # Check error
            err = np.linalg.norm(R - T)
            if err < 1e-9:
                U_out = U.tolist()
                V_out = V.tolist()
                W_out = W.tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # Local refinement if close
            if err < 1.0:
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    mat_idx = np.random.randint(3)
                    if mat_idx == 0:
                        r_idx, c_idx = np.random.randint(nm), np.random.randint(rank)
                        U[r_idx, c_idx] += np.random.choice([-0.5, 0.5])
                    elif mat_idx == 1:
                        r_idx, c_idx = np.random.randint(mp), np.random.randint(rank)
                        V[r_idx, c_idx] += np.random.choice([-0.5, 0.5])
                    else:
                        r_idx, c_idx = np.random.randint(pn), np.random.randint(rank)
                        W[r_idx, c_idx] += np.random.choice([-0.5, 0.5])
                    
                    R = np.einsum('as,bs,cs->abc', U, V, W)
                    err = np.linalg.norm(R - T)
                    if err < 1e-9:
                        U_out = U.tolist()
                        V_out = V.tolist()
                        W_out = W.tolist()
                        return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback: Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
