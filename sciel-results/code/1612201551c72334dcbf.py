import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    n, m, p = DIMENSIONS
    # T is a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _solve_linear_system(A, b):
    """Solve A x = b using least squares."""
    if A.shape[1] == 0:
        return np.zeros(0)
    # Use lstsq for overdetermined systems
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    num_a = n * m  # 6
    num_b = m * p  # 21
    num_c = p * n  # 14
    total_entries = num_a * num_b * num_c  # 1764
    
    # Precompute: for each a, the coefficient matrix for solving U[a,:]
    # When fixing V and W, for each a, we need to solve:
    # sum_c' U[a,c'] * (V[:,c'] * W[:,c'] flattened) = T[a,:,:] flattened
    # The coefficient matrix M_a has shape (num_b * num_c, rank)
    # M_a[(b,c), c'] = V[b, c'] * W[c, c']
    
    # We'll try ranks from a lower bound upward
    # Lower bound: max(num_a, num_b, num_c) = max(6, 21, 14) = 21
    # But we can try even lower
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank as fallback
    
    # Try ranks from 21 upward (or lower if we want to be aggressive)
    # Actually, let's try from a reasonable lower bound
    min_rank = max(num_a, num_b, num_c)  # 21
    
    # Let's try ranks from min_rank to min_rank + 5 or so
    for rank in range(min_rank, min_rank + 10):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random values
        np.random.seed(42)  # Reproducibility
        U = np.random.randn(num_a, rank) * 0.1
        V = np.random.randn(num_b, rank) * 0.1
        W = np.random.randn(num_c, rank) * 0.1
        
        # Iterate
        converged = False
        for it in range(200):
            if time.time() > deadline - 5:
                break
            
            # Solve for U: for each a, solve M_a * U[a,:] = T[a,:,:].flatten()
            new_U = np.zeros_like(U)
            for a in range(num_a):
                # M_a[(b,c), c'] = V[b, c'] * W[c, c']
                # Reshape V to (num_b, 1, rank) and W to (1, num_c, rank), broadcast
                M_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(num_b * num_c, rank)
                b_vec = T[a, :, :].reshape(-1)
                new_U[a, :] = _solve_linear_system(M_a, b_vec)
            U = new_U
            
            # Solve for V: for each b, solve M_b * V[b,:] = (U * T[:,:,b]).flatten()
            # When fixing U and W, for each b:
            # sum_c' V[b,c'] * (U[:,c'] * W[:,c'] with c fixed) = T[:,b,:]
            # M_b[(a,c), c'] = U[a, c'] * W[c, c']
            new_V = np.zeros_like(V)
            for b in range(num_b):
                M_b = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(num_a * num_c, rank)
                b_vec = T[:, b, :].reshape(-1)
                new_V[b, :] = _solve_linear_system(M_b, b_vec)
            V = new_V
            
            # Solve for W: for each c, solve M_c * W[c,:] = (U * V * T[:,:,c]).flatten()
            # When fixing U and V, for each c:
            # sum_c' W[c,c'] * (U[:,c'] * V[:,c'] with b fixed) = T[:,:,c]
            # M_c[(a,b), c'] = U[a, c'] * V[b, c']
            new_W = np.zeros_like(W)
            for c in range(num_c):
                M_c = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(num_a * num_b, rank)
                b_vec = T[:, :, c].reshape(-1)
                new_W[c, :] = _solve_linear_system(M_c, b_vec)
            W = new_W
        
        # Check if we found an exact solution
        # Efficient reconstruction: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can compute this using broadcasting and sums, or einsum
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        try:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            max_err = np.max(np.abs(T_recon - T))
            if max_err < 1e-10:
                best_rank = rank
                break
        except Exception:
            continue
    
    # If we didn't find a better rank, use schoolbook
    if best_rank == n * m * p:
        # Use schoolbook decomposition
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
    
    # Convert numpy arrays to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
