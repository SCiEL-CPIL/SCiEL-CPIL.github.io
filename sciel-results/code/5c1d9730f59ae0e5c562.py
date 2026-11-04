"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _is_half_integer(x, tol=1e-6):
    """Check if x is close to a half-integer."""
    r = 2 * x
    return abs(r - round(r)) < tol

def _snap_to_half_int(x, tol=1e-6):
    """Snap x to nearest half-integer if close."""
    r = 2 * x
    if abs(r - round(r)) < tol:
        return round(r) / 2.0
    return x

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = n, m, p
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check each half-integer
    for mat, name in [(U_arr, 'U'), (V_arr, 'V'), (W_arr, 'W')]:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not _is_half_integer(mat[i,j], tol):
                    return False
    
    # Verify tensor decomposition
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But actually the convention is: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = delta (contribution of A[i,j]*B[j,k] to C[i,k])
    # So T[i,j,k] = 1 for all valid (i,j,k)
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the W index is [k*n+i, r] per the task description.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - target) > tol:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the target tensor
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    # Start with schoolbook rank 24 as fallback
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(dim_U)]
    best_V = [[0] * best_rank for _ in range(dim_V)]
    best_W = [[0] * best_rank for _ in range(dim_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions using optimization
    # Strategy: for each target rank, try to find U,V,W with half-integer entries
    # that satisfy the tensor equation.
    
    # Use a structured initialization based on known decompositions
    # Try ranks from 14 up to 23
    
    for target_rank in range(14, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random half-integers in [-2, 2]
            rng = np.random.RandomState(attempt * 1000 + target_rank)
            
            U_init = np.zeros((dim_U, target_rank))
            V_init = np.zeros((dim_V, target_rank))
            W_init = np.zeros((dim_W, target_rank))
            
            # Structured initialization: use schoolbook-like patterns
            # Assign each (i,j,k) triple to a rank-1 component
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = (i * m + j) * p + k  # 0..23
                        if r < target_rank:
                            U_init[i*m+j, r] = 1
                            V_init[j*p+k, r] = 1
                            W_init[k*n+i, r] = 1
            
            # Add some random structure for remaining columns
            for r in range(target_rank):
                if U_init[:, r].sum() == 0:
                    # Random sparse initialization
                    idx_u = rng.randint(0, dim_U, size=2)
                    idx_v = rng.randint(0, dim_V, size=2)
                    idx_w = rng.randint(0, dim_W, size=2)
                    U_init[idx_u[0], r] = 1
                    U_init[idx_u[1], r] = 1
                    V_init[idx_v[0], r] = 1
                    V_init[idx_v[1], r] = 1
                    W_init[idx_w[0], r] = 1
                    W_init[idx_w[1], r] = 1
            
            # Gradient descent to minimize reconstruction error
            U_c = U_init.astype(float)
            V_c = V_init.astype(float)
            W_c = W_init.astype(float)
            
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient
                grad_U = np.zeros_like(U_c)
                grad_V = np.zeros_like(V_c)
                grad_W = np.zeros_like(W_c)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # residual = 1 - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                            residual = 1.0
                            for r in range(target_rank):
                                residual -= U_c[i*m+j, r] * V_c[j*p+k, r] * W_c[k*n+i, r]
                                grad_U[i*m+j, r] += residual * V_c[j*p+k, r] * W_c[k*n+i, r]
                                grad_V[j*p+k, r] += residual * U_c[i*m+j, r] * W_c[k*n+i, r]
                                grad_W[k*n+i, r] += residual * U_c[i*m+j, r] * V_c[j*p+k, r]
                
                U_c -= lr * grad_U
                V_c -= lr * grad_V
                W_c -= lr * grad_W
                
                # Periodic projection to half-integers
                if step % 200 == 0:
                    U_proj = np.vectorize(_snap_to_half_int)(U_c)
                    V_proj = np.vectorize(_snap_to_half_int)(V_c)
                    W_proj = np.vectorize(_snap_to_half_int)(W_c)
                    
                    # Check if projected version is valid
                    U_list = U_proj.tolist()
                    V_list = V_proj.tolist()
                    W_list = W_proj.tolist()
                    if _verify(U_list, V_list, W_list, n, m, p, tol=1e-4):
                        best_rank = target_rank
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        return {"U": best_U, "V": best_V, "W": best_W}
                    
                    # Update to projected values if error doesn't increase too much
                    U_c = U_proj
                    V_c = V_proj
                    W_c = W_proj
                
                # Adaptive learning rate
                if step % 500 == 0:
                    lr *= 0.5
            
            # Final check
            U_list = np.vectorize(_snap_to_half_int)(U_c).tolist()
            V_list = np.vectorize(_snap_to_half_int)(V_c).tolist()
            W_list = np.vectorize(_snap_to_half_int)(W_c).tolist()
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-4):
                best_rank = target_rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                return {"U": best_U, "V": best_V, "W": best_W}
    
    # Return best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
