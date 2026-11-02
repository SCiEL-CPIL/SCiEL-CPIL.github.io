import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    rank = 21
    
    # Strassen coefficients for 2x2 matrix multiplication
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]},
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]},
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]},
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]},
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
    ]
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U[idx][col] = sc['U'][idx]
            
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V[v_idx[idx]][col] = sc['V'][idx]
            
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W[w_idx[idx]][col] = sc['W'][idx]
    
    # Now try to find a linear dependency among the 21 rank-1 terms
    # Each rank-1 term is u_i ⊗ v_i ⊗ w_i
    # We want to find alpha_0, ..., alpha_20 (not all zero) such that
    # sum_i alpha_i * u_i ⊗ v_i ⊗ w_i = 0
    
    # The tensor has dimensions 4 x 12 x 12 = 576 entries
    # We have 21 rank-1 tensors, so we look for a null vector of the 576x21 matrix
    
    # Build the 576x21 matrix where each column is one flattened rank-1 tensor
    tensor_dim = 4 * 12 * 12  # 576
    M = np.zeros((tensor_dim, 21))
    
    for col in range(21):
        # Extract the rank-1 tensor for this column
        u_vec = np.array(U[:, col], dtype=float)
        v_vec = np.array(V[:, col], dtype=float)
        w_vec = np.array(W[:, col], dtype=float)
        
        # Flatten the outer product u ⊗ v ⊗ w
        # Index: (i, j, k) -> i * 12 * 12 + j * 12 + k
        for i in range(4):
            for j in range(12):
                for k in range(12):
                    M[i * 12 * 12 + j * 12 + k, col] = u_vec[i] * v_vec[j] * w_vec[k]
    
    # Find the null space of M (576x21 matrix)
    # If there's a non-trivial null vector, we have a dependency
    try:
        # SVD to find null space
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # The null space is spanned by rows of Vt corresponding to zero singular values
        # For a 576x21 matrix, Vt is 21x21
        # Singular values are in s (length 21)
        
        # Find indices where singular values are essentially zero
        tol = 1e-10
        null_indices = [i for i in range(21) if s[i] < tol]
        
        if len(null_indices) > 0:
            # We have a dependency! Use the last null vector
            null_vec = Vt[null_indices[-1], :]
            
            # Find the column with the largest |alpha| to eliminate
            max_idx = np.argmax(np.abs(null_vec))
            
            # Check if we can eliminate this column
            # The new decomposition: for each remaining column j != max_idx,
            # new_U[j] = U[j] + (alpha_j / alpha_max) * U[max_idx]
            # Similarly for V and W
            
            new_rank = 20
            new_U = [[0] * new_rank for _ in range(4)]
            new_V = [[0] * new_rank for _ in range(12)]
            new_W = [[0] * new_rank for _ in range(12)]
            
            new_col = 0
            for col in range(21):
                if col == max_idx:
                    continue
                scale = null_vec[col] / null_vec[max_idx]
                for i in range(4):
                    new_U[i][new_col] = U[i][col] + scale * U[i][max_idx]
                for j in range(12):
                    new_V[j][new_col] = V[j][col] + scale * V[j][max_idx]
                for k in range(12):
                    new_W[k][new_col] = W[k][col] + scale * W[k][max_idx]
                new_col += 1
            
            # Verify the decomposition
            # Check that the new decomposition is exact
            error = 0.0
            for i in range(4):
                for j in range(12):
                    for k in range(12):
                        # Compute the tensor entry from new decomposition
                        val = 0.0
                        for r in range(new_rank):
                            val += new_U[i][r] * new_V[j][r] * new_W[k][r]
                        # Compare with original
                        orig_val = 0.0
                        for r in range(21):
                            orig_val += U[i][r] * V[j][r] * W[k][r]
                        error += abs(val - orig_val)
            
            if error < 1e-8:
                return {"U": new_U, "V": new_V, "W": new_W}
            
    except Exception:
        pass
    
    # Fallback: return the rank-21 decomposition
    return {"U": U, "V": V, "W": W}
