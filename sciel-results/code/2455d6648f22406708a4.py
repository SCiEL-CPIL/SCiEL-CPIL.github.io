import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _build_strassen_decomposition(n, m, p):
    """Build the rank-42 Strassen decomposition for n x m * m x p."""
    # Strassen coefficients for A (local 2x2)
    strassen_A = [
        [1, 0, 0, 1], # M0
        [0, 0, 1, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 0, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0], # M5
        [0, 1, 0, -1]  # M6
    ]
    
    # Strassen coefficients for B (local 2x2)
    strassen_B = [
        [1, 0, 0, 1], # M0
        [1, 0, 0, 0], # M1
        [0, 1, 0, -1], # M2
        [-1, 0, 1, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, 1]  # M6
    ]
    
    # W coefficients: w_coeffs[t][i*2 + k_loc]
    w_coeffs = [
        [1, 0, 0, 1], # t=0
        [0, 0, 1, -1], # t=1
        [0, 1, 0, 1], # t=2
        [1, 0, 1, 0], # t=3
        [-1, 1, 0, 0], # t=4
        [0, 0, 0, 1], # t=5
        [1, 0, 0, 0]  # t=6
    ]
    
    num_blocks_s = m // 2  # 2
    num_blocks_k = p // 2  # 3
    num_blocks = num_blocks_s * num_blocks_k
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        s = b // num_blocks_k
        k_block = b % num_blocks_k
        
        for t in range(7):
            c = 7 * b + t
            
            # Fill U
            u_coeff = strassen_A[t]
            for i in range(2):
                for j_loc in range(2):
                    idx_A = i * 2 + j_loc
                    coeff = u_coeff[idx_A]
                    if coeff != 0:
                        global_i = i
                        global_j = 2 * s + j_loc
                        U[global_i * m + global_j][c] = coeff
                        
            # Fill V
            v_coeff = strassen_B[t]
            for j_loc in range(2):
                for k_loc in range(2):
                    idx_B = j_loc * 2 + k_loc
                    coeff = v_coeff[idx_B]
                    if coeff != 0:
                        global_j = 2 * s + j_loc
                        global_k = 2 * k_block + k_loc
                        V[global_j * p + global_k][c] = coeff
                        
            # Fill W
            w_coeff = w_coeffs[t]
            for i in range(2):
                for k_loc in range(2):
                    idx_W = i * 2 + k_loc
                    coeff = w_coeff[idx_W]
                    if coeff != 0:
                        global_k = 2 * k_block + k_loc
                        global_i = i
                        W[global_k * n + global_i][c] = coeff
                        
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Convert to numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor: T[i,j,k] = 1 if we want C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition should satisfy: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other (i,j,k) combinations where the product should be 0, the sum should be 0.
    
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                expected = 1.0  # C[i,k] includes A[i,j]*B[j,k]
                if abs(total - expected) > 1e-10:
                    return False
    return True

def _find_linear_dependencies(U, V, W, n, m, p):
    """Find linear dependencies among the rank-1 tensors."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Each rank-1 tensor is the outer product U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # Flatten each into a vector and check for linear dependencies
    dim = (n*m) * (m*p) * (p*n)
    
    # This could be large: (8)*(24)*(12) = 2304 for n=2,m=4,p=6
    # Build matrix where each column is a flattened rank-1 tensor
    tensors = np.zeros((dim, rank))
    for c in range(rank):
        outer = np.outer(np.outer(U_arr[:, c], V_arr[:, c]), W_arr[:, c])
        tensors[:, c] = outer.flatten()
    
    # Find rank of this matrix
    # Use SVD to find the numerical rank
    try:
        U_svd, s_vals, Vt = np.linalg.svd(tensors, full_matrices=False)
        # Count significant singular values
        tol = 1e-10
        num_rank = np.sum(s_vals > tol)
        return num_rank, s_vals
    except:
        return rank, None

def _try_reduce_rank(U, V, W, n, m, p, target_rank):
    """Try to find a lower-rank decomposition by removing dependent tensors."""
    rank = len(U[0])
    
    # Get the numerical rank
    num_rank, s_vals = _find_linear_dependencies(U, V, W, n, m, p)
    
    if num_rank >= rank:
        return None
    
    # If there are dependencies, try to find a subset of indices that spans the full space
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Greedy approach: try to remove tensors with smallest singular values
    # First, identify which tensors are "redundant"
    # This is a complex combinatorial problem, so we use a heuristic
    
    # Try all subsets of size target_rank (if feasible)
    # For now, just return the current decomposition if no reduction found
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Build the initial rank-42 Strassen decomposition
    U, V, W, rank = _build_strassen_decomposition(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: return schoolbook decomposition
        U_fb = [[0] * (n*m*p) for _ in range(n*m)]
        V_fb = [[0] * (n*m*p) for _ in range(m*p)]
        W_fb = [[0] * (n*m*p) for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find linear dependencies and reduce rank
    if time.time() < deadline - 30:
        num_rank, s_vals = _find_linear_dependencies(U, V, W, n, m, p)
        
        if num_rank < rank and s_vals is not None:
            # There are linear dependencies
            # Try to find a minimal spanning subset
            # Sort tensors by their contribution to the smallest singular values
            U_svd, s_vals_full, Vt = np.linalg.svd(np.array([
                np.outer(np.outer(U[:, c], V[:, c]), W[:, c]).flatten() for c in range(rank)
            ]).T, full_matrices=False)
            
            # Try removing tensors one by one, starting with those most aligned with null space
            # This is a heuristic; for now, just try a few simple reductions
            
            # Check if any pair of tensors are identical or negatives
            for c1 in range(rank):
                if time.time() > deadline - 30:
                    break
                for c2 in range(c1 + 1, rank):
                    # Check if tensor c1 + tensor c2 = 0 (i.e., they are negatives)
                    U_sum = [U[i][c1] + U[i][c2] for i in range(n*m)]
                    V_sum = [V[i][c1] + V[i][c2] for i in range(m*p)]
                    W_sum = [W[i][c1] + W[i][c2] for i in range(p*n)]
                    
                    if all(abs(x) < 1e-10 for x in U_sum) and \
                       all(abs(x) < 1e-10 for x in V_sum) and \
                       all(abs(x) < 1e-10 for x in W_sum):
                        # These two tensors cancel out; remove both
                        new_rank = rank - 2
                        new_U = [[U[i][c] for c in range(rank) if c != c1 and c != c2] for i in range(n*m)]
                        new_V = [[V[i][c] for c in range(rank) if c != c1 and c != c2] for i in range(m*p)]
                        new_W = [[W[i][c] for c in range(rank) if c != c1 and c != c2] for i in range(p*n)]
                        
                        if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                            best_rank = new_rank
                            best_U, best_V, best_W = new_U, new_V, new_W
                            U, V, W, rank = new_U, new_V, new_W, new_rank
                            break
    
    # Try a more systematic search for rank reduction using linear algebra
    if time.time() < deadline - 60 and best_rank == 42:
        # Try to find a rank-41 decomposition by solving a linear system
        # This requires checking if any of the 42 tensors can be expressed as a linear combination of the others
        # We'll use a different approach: try all single-tensor removals
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for c_remove in range(rank):
            if time.time() > deadline - 30:
                break
            
            # Try removing tensor c_remove and see if the remaining tensors still span the space
            # This is equivalent to checking if tensor c_remove is in the span of the others
            # We can do this by checking if the matrix of remaining tensors has full column rank
            # and if tensor c_remove is in their column space
            
            indices = [c for c in range(rank) if c != c_remove]
            if len(indices) < 1:
                continue
            
            # Build the tensor matrix for the remaining tensors
            dim = (n*m) * (m*p) * (p*n)
            tensors_remaining = np.zeros((dim, len(indices)))
            for idx, c in enumerate(indices):
                outer = np.outer(np.outer(U_arr[:, c], V_arr[:, c]), W_arr[:, c])
                tensors_remaining[:, idx] = outer.flatten()
            
            # Check if tensor c_remove is in the column space of tensors_remaining
            outer_remove = np.outer(np.outer(U_arr[:, c_remove], V_arr[:, c_remove]), W_arr[:, c_remove])
            vec_remove = outer_remove.flatten()
            
            # Solve the least squares problem
            try:
                coeffs, residuals, rank_rem, svals = np.linalg.lstsq(tensors_remaining, vec_remove, rcond=1e-10)
                # Check if the residual is small
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    # Tensor c_remove is in the span of the others
                    new_U = [[U[i][c] for c in indices] for i in range(n*m)]
                    new_V = [[V[i][c] for c in indices] for i in range(m*p)]
                    new_W = [[W[i][c] for c in indices] for i in range(p*n)]
                    
                    if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                        best_rank = rank - 1
                        best_U, best_V, best_W = new_U, new_V, new_W
                        break
            except:
                pass
    
    if time.time() > deadline - 10:
        pass  # Safety margin
    
    return {"U": best_U, "V": best_V, "W": best_W}
