"""Matrix multiplication 2x3 by 3x4: find rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Uses Simulated Annealing to search for a rank-19 decomposition with half-integer entries.
    Falls back to schoolbook rank 24 if not found within the time limit.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 4
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    target_rank = 19

    # Build the list of constraints: (idx_U, idx_V, idx_W)
    # The tensor entry T[idx_U, idx_V, idx_W] must be 1.
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                constraints.append((idx_u, idx_v, idx_w))
    
    num_constraints = len(constraints) # 24

    # Pre-compute indices for vectorized residual calculation
    # We have C constraints. For each constraint c, we have indices u_c, v_c, w_c.
    # We want to compute sum_r U[u_c, r] * V[v_c, r] * W[w_c, r] for all c.
    # Let U_idx be shape (C, R), V_idx be (C, R), W_idx be (C, R).
    # Then residuals = sum_r (U_idx * V_idx * W_idx) - 1.
    
    U_idx = np.array([c[0] for c in constraints]) # shape (C,)
    V_idx = np.array([c[1] for c in constraints]) # shape (C,)
    W_idx = np.array([c[2] for c in constraints]) # shape (C,)

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify_solution(U, V, W, rank):
        """
        Verify if U, V, W of given rank exactly decompose the tensor.
        Returns True if sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r] == 1 for all constraints.
        """
        # Vectorized check
        # U[U_idx, :] has shape (C, R)
        # V[V_idx, :] has shape (C, R)
        # W[W_idx, :] has shape (C, R)
        products = U[U_idx, :] * V[V_idx, :] * W[W_idx, :]
        sums = np.sum(products, axis=1)
        # Check if all sums are close to 1.0
        return np.allclose(sums, 1.0, atol=1e-9)

    def get_residue(U, V, W):
        """Calculate the sum of squared errors for the current solution."""
        products = U[U_idx, :] * V[V_idx, :] * W[W_idx, :]
        sums = np.sum(products, axis=1)
        residuals = sums - 1.0
        return np.sum(residuals * residuals)

    def clean_output(U, V, W):
        """Convert numpy arrays to lists of ints/floats for the required format."""
        def clean_matrix(M):
            M_clean = []
            for row in M:
                clean_row = []
                for val in row:
                    # Check if it's an integer
                    if abs(val - round(val)) < 1e-9:
                        clean_row.append(int(round(val)))
                    else:
                        # It's a half-integer (or close to it)
                        clean_row.append(float(val))
                M_clean.append(clean_row)
            return M_clean
        
        return {
            "U": clean_matrix(U),
            "V": clean_matrix(V),
            "W": clean_matrix(W)
        }

    # --- Simulated Annealing Search ---
    
    # Allowed values for entries
    # We expect small integers or half-integers. 
    # Based on known decompositions, values are usually in {-1.5, -1, -0.5, 0, 0.5, 1, 1.5}
    # Let's start with a slightly wider set to allow exploration, then narrow if needed.
    # However, for rank 19, values are typically small.
    allowed_values = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    
    rng = np.random.default_rng(42)
    
    # Initialize U, V, W with random allowed values
    # It's often better to start with zeros and a few non-zeros, or random sparse.
    # Let's try random initialization from the allowed set, but sparsified.
    # A fully dense random matrix is likely far from the solution.
    # Let's initialize with 0s and set ~20% of entries to random allowed values.
    
    def init_sparse_matrix(rows, cols, density=0.2):
        M = np.zeros((rows, cols))
        mask = rng.random((rows, cols)) < density
        # Assign random values from allowed set to masked positions
        random_vals = rng.choice(allowed_values, size=(rows, cols))
        M[mask] = random_vals[mask]
        return M

    U = init_sparse_matrix(dim_U, target_rank)
    V = init_sparse_matrix(dim_V, target_rank)
    W = init_sparse_matrix(dim_W, target_rank)
    
    current_error = get_residue(U, V, W)
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # SA Parameters
    T = 10.0  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    
    step_count = 0
    no_improve_count = 0
    
    while time.time() < DEADLINE:
        if T < T_min:
            break
            
        # Select a random matrix to perturb
        mat_idx = rng.integers(0, 3)
        if mat_idx == 0:
            M = U
        elif mat_idx == 1:
            M = V
        else:
            M = W
            
        # Select a random row and column
        r = rng.integers(0, M.shape[0])
        c = rng.integers(0, M.shape[1])
        
        old_val = M[r, c]
        
        # Generate a new value
        # 80% chance: pick a new random value from allowed set
        # 20% chance: keep old value (to allow "null" moves or just randomness)
        if rng.random() < 0.8:
            new_val = rng.choice(allowed_values)
        else:
            new_val = old_val
            
        if new_val == old_val:
            continue
            
        # Apply change
        M[r, c] = new_val
        
        # Calculate new error
        if mat_idx == 0:
            new_error = get_residue(U, V, W)
        elif mat_idx == 1:
            new_error = get_residue(U, V, W)
        else:
            new_error = get_residue(U, V, W)
            
        delta = new_error - current_error
        
        # Acceptance probability
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept move
            current_error = new_error
            if new_error < best_error:
                best_error = new_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject move
            M[r, c] = old_val
            no_improve_count += 1
            
        # Cool down
        T *= cooling_rate
        
        # Check for exact solution periodically
        if step_count % 100 == 0 and best_error < 1e-6:
            if verify_solution(best_U, best_V, best_W, target_rank):
                return clean_output(best_U, best_V, best_W)
                
        step_count += 1

    # Final check on the best found solution
    if best_error < 1e-6 and verify_solution(best_U, best_V, best_W, target_rank):
        return clean_output(best_U, best_V, best_W)
        
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
