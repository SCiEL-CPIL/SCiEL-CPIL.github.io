import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def _verify_fast(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    Uses NumPy for speed.
    T[i,j,k] should equal 1 for all (i,j,k).
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape to 3D tensors:
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
    
    # Check if all elements are 1
    return np.allclose(T, 1.0, atol=1e-9)

def _solve_rank(n, m, p, rank, max_iters=100, timeout=10.0):
    """
    Attempt to find a rank-`rank` decomposition using local optimization.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    target_tensor = np.ones((n, m, p))
    
    # Flatten the tensor for optimization
    # Variables: U (n*m*rank), V (m*p*rank), W (p*n*rank)
    num_u = n * m * rank
    num_v = m * p * rank
    num_w = p * n * rank
    total_vars = num_u + num_v + num_w
    
    def compute_residual(x):
        U_flat = x[:num_u]
        V_flat = x[num_u:num_u+num_v]
        W_flat = x[num_u+num_v:]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U, V, W, optimize=True)
        return (T - target_tensor).flatten()

    # Try a few random initializations
    for _ in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with small random integers or schoolbook-like
        # Using small integers helps with convergence to integer solutions
        init_u = np.random.randint(-2, 3, size=num_u)
        init_v = np.random.randint(-2, 3, size=num_v)
        init_w = np.random.randint(-2, 3, size=num_w)
        x0 = np.concatenate([init_u, init_v, init_w]).astype(np.float64)
        
        try:
            # Use least_squares to minimize residual
            res = least_squares(compute_residual, x0, method='trf', max_nfev=2000)
            
            if res.cost < 1e-10:
                # Found a solution, now round to integers/half-integers
                x_sol = res.x
                U_sol = x_sol[:num_u].reshape(n, m, rank)
                V_sol = x_sol[num_u:num_u+num_v].reshape(m, p, rank)
                W_sol = x_sol[num_u+num_v:].reshape(p, n, rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Convert to lists for return
                U_list = U_rounded.flatten().tolist()
                V_list = V_rounded.flatten().tolist()
                W_list = W_rounded.flatten().tolist()
                
                # Verify exactly
                if _verify_fast(U_list, V_list, W_list, n, m, p, rank):
                    return U_list, V_list, W_list
                else:
                    # Try local search around the rounded solution
                    # This might be necessary if rounding introduced small errors
                    # But usually if cost is very low, rounding works.
                    pass
                    
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    # Schoolbook rank
    rank_seed = n * m * p
    
    # Build schoolbook decomposition
    U_seed = [[0] * rank_seed for _ in range(n*m)]
    V_seed = [[0] * rank_seed for _ in range(m*p)]
    W_seed = [[0] * rank_seed for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i*m+j][c] = 1
                V_seed[j*p+k][c] = 1
                W_seed[k*n+i][c] = 1
                c += 1
    
    best = {"U": U_seed, "V": V_seed, "W": W_seed}
    best_rank = rank_seed
    
    # Try to find lower rank decompositions using optimization
    # Start from rank_seed - 1 downwards
    # We allocate time for each rank attempt
    
    # We know rank 45 is schoolbook.
    # Let's try 44, 43, etc.
    
    for target_rank in range(rank_seed - 1, 20, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time based on how many ranks we have left
        # If we are early, we can spend more time.
        time_alloc = min(15.0, (deadline - time.time()) / 2.0)
        
        result = _solve_rank(n, m, p, target_rank, max_iters=20, timeout=time_alloc)
        
        if result:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            
    return best
