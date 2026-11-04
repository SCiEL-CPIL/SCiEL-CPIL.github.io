"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Schoolbook fallback (Rank 40)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    # Attempt to find a lower rank decomposition using Simulated Annealing
    # We target rank 30 (a reasonable lower bound estimate, though exactness is hard)
    target_rank = 30
    
    # Reshape dimensions for easier tensor manipulation
    # T[i, j, k] = 1 if we are computing C[i, k] from A[i, j] * B[j, k]
    # Dimensions: i in [0, n), j in [0, m), k in [0, p)
    
    # We represent the tensor T of shape (n, m, p)
    # Decomposition: T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: The task asks for U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # U_flat[i*m+j, r], V_flat[j*p+k, r], W_flat[k*n+i, r]
    
    # Objective: Minimize || T - T_decomposed ||_F^2
    # T is sparse, mostly zeros except T[i, j, k] = 1 for all i, j, k.
    # Wait, the multiplication tensor T_{ijk} is 1 if the term A_{ij}B_{jk} contributes to C_{ik}.
    # For standard matrix multiplication C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor is T[i, j, k] = 1 for all i, j, k.
    # The decomposition is T[i, j, k] = sum_r U[i, m+j, r] * V[j, p+k, r] * W[k, n+i, r]
    
    # Let's verify indices.
    # U has shape (n*m, rank). Index by (i, j) -> i*m + j.
    # V has shape (m*p, rank). Index by (j, k) -> j*p + k.
    # W has shape (p*n, rank). Index by (k, i) -> k*n + i.
    
    # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Summing over r gives the coefficient for A[i,j]B[j,k] in C[i,k].
    # We want this sum to be 1 for all i, j, k.
    
    # So the target tensor T is all ones of shape (n, m, p).
    
    def compute_error(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        
        # Compute T_decomp[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
        # This is a bit tricky with standard einsum because of the cyclic indices.
        # Let's use einsum.
        # U: i j r
        # V: j k r
        # W: k i r
        # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # First, combine U and V: UV[i, j, k, r] = U[i, j, r] * V[j, k, r]
        # Then contract with W: T[i, j, k] = sum_r UV[i, j, k, r] * W[k, i, r]
        
        # Efficient way:
        # E1 = np.einsum('ijr,jkr->ijk r', U_r, V_r) # Shape (n, m, p, R)
        # T = np.einsum('ijkr,ki r->ijk', E1, W_r)   # Shape (n, m, p)
        
        # Let's check memory. n=2, m=4, p=5, R=30.
        # E1 size: 2*4*5*30 = 1200. Very small.
        
        E1 = np.einsum('ijr,jkr->ijk r', U_r, V_r)
        T_decomp = np.einsum('ijkr,ki r->ijk', E1, W_r)
        
        # Target is all ones
        diff = T_decomp - 1.0
        return np.sum(diff * diff)

    # Initialize with random values
    # We can start from a random perturbation of the schoolbook, or just random.
    # Random initialization often works well with SA for small problems.
    
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    current_error = compute_error(U, V, W)
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Simulated Annealing Parameters
    T_current = 100.0
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    max_iter = 50000
    
    # Pre-allocate arrays for speed
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    
    # We will perturb one element at a time
    total_elements = U_flat.size + V_flat.size + W_flat.size
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Check if we have found an exact solution (error < 1e-10)
        if best_error < 1e-10:
            break
            
        # Choose a random parameter to perturb
        idx = np.random.randint(total_elements)
        
        # Determine which array and index
        if idx < U_flat.size:
            arr_flat = U_flat
            arr = U
            arr_idx = idx
            arr_name = 'U'
        elif idx < U_flat.size + V_flat.size:
            arr_flat = V_flat
            arr = V
            arr_idx = idx - U_flat.size
            arr_name = 'V'
        else:
            arr_flat = W_flat
            arr = W
            arr_idx = idx - U_flat.size - V_flat.size
            arr_name = 'W'
            
        # Save old value
        old_val = arr_flat[arr_idx]
        
        # Perturb
        step_size = 0.1
        new_val = old_val + np.random.randn() * step_size
        arr_flat[arr_idx] = new_val
        
        # Compute new error
        new_error = compute_error(arr, V if arr_name != 'V' else V, W if arr_name != 'W' else W)
        # Note: compute_error takes U, V, W. Since only one changed, we pass the others directly.
        # But compute_error reshapes them, so it's fine.
        
        delta_e = new_error - current_error
        
        # Accept if better, or with probability exp(-delta_e / T)
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T_current):
            current_error = new_error
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # If we found a very low error, we might want to reduce step size or cool faster
        else:
            # Revert
            arr_flat[arr_idx] = old_val
            
        # Cool down
        T_current *= alpha
        if T_current < T_min:
            T_current = T_min

    # Check if we found a valid lower rank solution
    if best_error < 1e-10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fall back to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
