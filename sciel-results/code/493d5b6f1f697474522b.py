"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Lower bound for rank
    # Rank(3x3x3) = 20. Rank(3x3x5) is likely around 22-25.
    # We search from 20 up to 45 (schoolbook).
    
    for rank in range(20, 46):
        # Try to find a solution with this rank
        # Use discrete local search
        
        # We will try a few random restarts
        found = False
        for restart in range(20):
            np.random.seed(42 + rank * 100 + restart)
            
            # Initialize with random integers -1, 0, 1
            U = np.random.choice([-1, 0, 1], size=(n * m, rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, rank))
            
            # Compute T
            # Reshape for einsum
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            
            T = np.einsum('ijr,jkr,kin->ijk', U_r, V_r, W_r)
            target = np.ones((n, m, p))
            error = np.sum((T - target) ** 2)
            
            if error < 1e-10:
                found = True
                break
            
            # Local search
            max_steps = 5000
            for step in range(max_steps):
                if error < 1e-10:
                    found = True
                    break
                
                # Pick a random variable to change
                # 0 for U, 1 for V, 2 for W
                var_type = np.random.randint(3)
                
                if var_type == 0:
                    # Change U[i*m+j, r]
                    idx_u = np.random.randint(n * m)
                    r = np.random.randint(rank)
                    i = idx_u // m
                    j = idx_u % m
                    
                    old_val = U[idx_u, r]
                    # Pick new value
                    possible = [v for v in [-1, 0, 1] if v != old_val]
                    new_val = np.random.choice(possible)
                    delta = new_val - old_val
                    
                    if delta == 0:
                        continue
                    
                    # Calculate change in T
                    # T[i, j, k] += delta * V[j*p+k, r] * W[k*n+i, r]
                    # V indices: j*p + k for k in 0..p-1
                    # W indices: k*n + i for k in 0..p-1
                    
                    V_slice = V[j * p : j * p + p, r]
                    W_indices = np.arange(p) * n + i
                    W_slice = W[W_indices, r]
                    
                    delta_T = delta * V_slice * W_slice
                    
                    # Calculate error change
                    T_old_slice = T[i, j, :]
                    T_new_slice = T_old_slice + delta_T
                    
                    error_old_slice = np.sum((T_old_slice - 1) ** 2)
                    error_new_slice = np.sum((T_new_slice - 1) ** 2)
                    
                    delta_error = error_new_slice - error_old_slice
                    
                    if delta_error < 0:
                        # Accept
                        U[idx_u, r] = new_val
                        T[i, j, :] = T_new_slice
                        error += delta_error
                        
                elif var_type == 1:
                    # Change V[j*p+k, r]
                    idx_v = np.random.randint(m * p)
                    r = np.random.randint(rank)
                    j = idx_v // p
                    k = idx_v % p
                    
                    old_val = V[idx_v, r]
                    possible = [v for v in [-1, 0, 1] if v != old_val]
                    new_val = np.random.choice(possible)
                    delta = new_val - old_val
                    
                    if delta == 0:
                        continue
                    
                    # T[i, j, k] += delta * U[i*m+j, r] * W[k*n+i, r]
                    # Affected i: 0..n-1
                    # U indices: i*m + j
                    # W indices: k*n + i
                    
                    U_indices = np.arange(n) * m + j
                    U_slice = U[U_indices, r]
                    
                    W_indices = np.arange(n) # k*n + i? No, W is k*n + i.
                    # Here k is fixed. i varies.
                    # W index for fixed k, varying i: k*n + i.
                    W_slice = W[k * n : k * n + n, r]
                    
                    delta_T = delta * U_slice * W_slice
                    
                    # T[i, j, k] for all i
                    T_old_slice = T[:, j, k]
                    T_new_slice = T_old_slice + delta_T
                    
                    error_old_slice = np.sum((T_old_slice - 1) ** 2)
                    error_new_slice = np.sum((T_new_slice - 1) ** 2)
                    
                    delta_error = error_new_slice - error_old_slice
                    
                    if delta_error < 0:
                        V[idx_v, r] = new_val
                        T[:, j, k] = T_new_slice
                        error += delta_error
                        
                else:
                    # Change W[k*n+i, r]
                    idx_w = np.random.randint(p * n)
                    r = np.random.randint(rank)
                    k = idx_w // n
                    i = idx_w % n
                    
                    old_val = W[idx_w, r]
                    possible = [v for v in [-1, 0, 1] if v != old_val]
                    new_val = np.random.choice(possible)
                    delta = new_val - old_val
                    
                    if delta == 0:
                        continue
                    
                    # T[i, j, k] += delta * U[i*m+j, r] * V[j*p+k, r]
                    # Affected j: 0..m-1
                    # U indices: i*m + j
                    # V indices: j*p + k
                    
                    U_indices = i * m + np.arange(m)
                    U_slice = U[U_indices, r]
                    
                    V_indices = np.arange(m) * p + k
                    V_slice = V[V_indices, r]
                    
                    delta_T = delta * U_slice * V_slice
                    
                    # T[i, j, k] for all j
                    T_old_slice = T[i, :, k]
                    T_new_slice = T_old_slice + delta_T
                    
                    error_old_slice = np.sum((T_old_slice - 1) ** 2)
                    error_new_slice = np.sum((T_new_slice - 1) ** 2)
                    
                    delta_error = error_new_slice - error_old_slice
                    
                    if delta_error < 0:
                        W[idx_w, r] = new_val
                        T[i, :, k] = T_new_slice
                        error += delta_error
            
            if found:
                break
        
        if found:
            # Verify and return
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            T_final = np.einsum('ijr,jkr,kin->ijk', U_r, V_r, W_r)
            
            if np.allclose(T_final, np.ones((n, m, p)), atol=1e-10):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
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
