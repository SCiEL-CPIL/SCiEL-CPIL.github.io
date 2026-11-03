"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
n, m, p = DIMENSIONS
N_U = n * m  # 10
N_V = m * p  # 25
N_W = p * n  # 10

# EVOLVE-BLOCK-START
def _build_tensor():
    """
    Build the matrix multiplication tensor T of shape (N_U, N_V, N_W).
    T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    """
    T = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _structured_init(rank, seed):
    """
    Initialize U, V, W with structured random values.
    U: (N_U, rank)
    V: (N_V, rank)
    W: (N_W, rank)
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(N_U, rank) * 0.1
    V = rng.randn(N_V, rank) * 0.1
    W = rng.randn(N_W, rank) * 0.1
    
    # Add a bit of structure: ensure no rows are zero
    for i in range(N_U):
        U[i, i % rank] += 1.0
    for i in range(N_V):
        V[i, i % rank] += 1.0
    for i in range(N_W):
        W[i, i % rank] += 1.0
        
    return U, V, W

def _als_step(U, V, W, T, max_iter=100, tol=1e-12):
    """
    Run Alternating Least Squares to minimize ||T - sum_r U[:,r] V[:,r]^T W[:,r]^T||_F
    Actually, the tensor T is flattened. 
    Residual R = T - sum_r U[:,r] V[:,r]^T W[:,r]^T is not quite right for the tensor form.
    
    The tensor decomposition is:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    For fixed V, W, we solve for U:
    T[a, b, c] = sum_r U[a, r] * (V[b, r] * W[c, r])
    This is a linear system for each 'r' dimension? No, it's a system for U.
    Let M_{ab, c} = T[a, b, c].
    For a fixed 'r', the term is U[a, r] V[b, r] W[c, r].
    This is bilinear.
    
    Standard ALS for CP decomposition:
    Fix V, W. Solve for U.
    The equation is: T = sum_r U[:,r] * (V[:,r] * W[:,r])^T ? No.
    The tensor T is 3-way.
    Let's flatten T into a matrix T_{(N_U * N_V), N_W} or similar?
    
    Better approach:
    For fixed V, W, the model is:
    T[a, b, c] = sum_r U[a, r] * K_r[b, c]
    where K_r[b, c] = V[b, r] * W[c, r].
    
    This is a linear least squares problem for U.
    We can vectorize.
    Let T_mat be of shape (N_U, N_V * N_W).
    T_mat[a, b*N_W + c] = T[a, b, c].
    
    Let K be of shape (N_V * N_W, rank).
    K[b*N_W + c, r] = V[b, r] * W[c, r].
    
    Then T_mat = U @ K^T.
    So U = T_mat @ K^{-1} (least squares).
    
    Similarly for V and W.
    """
    # Precompute flattened T
    T_flat_U = T.reshape(N_U, N_V * N_W) # Shape (10, 250)
    T_flat_V = T.reshape(N_V, N_U * N_W) # Shape (25, 100)
    T_flat_W = T.reshape(N_W, N_U * N_V) # Shape (10, 250)
    
    residual = np.inf
    for it in range(max_iter):
        # Solve for U
        # K_U shape: (N_V * N_W, rank)
        K_U = np.zeros((N_V * N_W, rank), dtype=np.float64)
        for b in range(N_V):
            for c in range(N_W):
                K_U[b * N_W + c, :] = V[b, :] * W[c, :]
        
        # T_flat_U = U @ K_U^T  => U = T_flat_U @ K_U^T^+
        # Use least squares
        try:
            U_new, _, _, _ = np.linalg.lstsq(K_U.T, T_flat_U.T, rcond=None)
            U_new = U_new.T
        except np.linalg.LinAlgError:
            break
        
        # Solve for V
        # K_V shape: (N_U * N_W, rank)
        K_V = np.zeros((N_U * N_W, rank), dtype=np.float64)
        for a in range(N_U):
            for c in range(N_W):
                K_V[a * N_W + c, :] = U[a, :] * W[c, :]
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(K_V.T, T_flat_V.T, rcond=None)
            V_new = V_new.T
        except np.linalg.LinAlgError:
            break
            
        # Solve for W
        # K_W shape: (N_U * N_V, rank)
        K_W = np.zeros((N_U * N_V, rank), dtype=np.float64)
        for a in range(N_U):
            for b in range(N_V):
                K_W[a * N_V + b, :] = U[a, :] * V[b, :]
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(K_W.T, T_flat_W.T, rcond=None)
            W_new = W_new.T
        except np.linalg.LinAlgError:
            break
            
        U, V, W = U_new, V_new, W_new
        
        # Calculate residual
        # Reconstruct T
        T_rec = np.zeros_like(T)
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        residual = np.linalg.norm(T - T_rec) / np.linalg.norm(T)
        
        if residual < tol:
            break
            
    return U, V, W, residual

def _polish(U, V, W, T, max_iter=500, tol=1e-14):
    """High precision polish using Gauss-Newton or tighter ALS."""
    return _als_step(U, V, W, T, max_iter=max_iter, tol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    T = _build_tensor()
    
    best_result = None
    best_rank = n * m * p  # 50
    best_residual = np.inf
    
    # Try ranks from lower bound upwards
    # Lower bound is max(N_U, N_V, N_W) = 25
    min_rank = max(N_U, N_V, N_W)
    
    # We will try to find a decomposition for rank r.
    # If we find one with residual < 1e-10, we accept it.
    
    # Start with schoolbook to have a valid fallback
    U_school = np.zeros((N_U, best_rank))
    V_school = np.zeros((N_V, best_rank))
    W_school = np.zeros((N_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    best_result = (U_school, V_school, W_school)
    best_residual = 0.0
    
    # Search for lower rank
    for rank in range(min_rank, best_rank + 1):
        if time.time() > deadline - 30:
            break
            
        found = False
        # Try multiple seeds for this rank
        for seed in range(50):
            if time.time() > deadline - 10:
                break
                
            U, V, W = _structured_init(rank, seed)
            
            # Run ALS
            U, V, W, res = _als_step(U, V, W, T, max_iter=200, tol=1e-12)
            
            if res < 1e-10:
                # Polish
                U, V, W, res = _polish(U, V, W, T, max_iter=100, tol=1e-14)
                if res < 1e-10:
                    best_result = (U, V, W)
                    best_rank = rank
                    best_residual = res
                    found = True
                    break
        
        if found:
            # If we found a low rank, we might try to go even lower? 
            # The loop goes upwards, so if we found rank r, we break because we want minimal rank.
            # Wait, the loop is `for rank in range(min_rank, best_rank+1)`.
            # So it tries min_rank, then min_rank+1, etc.
            # If we succeed at min_rank, we break out of the seed loop, then out of the rank loop?
            # Yes, because we want the *minimal* rank.
            break
    
    U, V, W = best_result
    
    # Convert to list format
    result = {}
    for key, mat in [('U', U), ('V', V), ('W', W)]:
        converted = []
        for row in mat:
            new_row = []
            for val in row:
                # Check if close to integer
                if abs(val - round(val)) < 1e-8:
                    new_row.append(int(round(val)))
                else:
                    # Check if close to half-integer
                    half_val = round(val * 2) / 2.0
                    if abs(val - half_val) < 1e-8:
                        new_row.append(half_val)
                    else:
                        new_row.append(float(val))
            converted.append(new_row)
        result[key] = converted
    
    return result
# EVOLVE-BLOCK-END
