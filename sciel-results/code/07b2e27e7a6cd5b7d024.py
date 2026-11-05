import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Start with schoolbook rank-72 decomposition
    total_rank = 72
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_U = U_np
    best_V = V_np
    best_W = W_np
    best_rank = total_rank
    
    # Try to find rank reduction by combining columns
    # Strategy: randomly select subsets of columns, form linear combinations,
    # check if the resulting tensor still matches
    
    np.random.seed(42)
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k (since every term contributes)
    # Actually, for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has T[i,j,k] = 1 for all valid i,j,k
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition correctly represents the multiplication tensor."""
        # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j, :]
                    v_val = V_mat[j * p + k, :]
                    w_val = W_mat[k * n + i, :]
                    dot = np.sum(u_val * v_val * w_val)
                    if abs(dot - 1.0) > 1e-9:
                        return False
        return True
    
    # Try random linear combinations to reduce rank
    # We'll try to replace pairs of columns with a single combined column
    
    attempts = 0
    max_attempts = 5000
    
    while attempts < max_attempts and (time.time() - start_time) < deadline:
        attempts += 1
        
        # Pick two random distinct columns
        c1, c2 = np.random.choice(best_rank, size=2, replace=False)
        
        # Try to combine them: new_col = a * col1 + b * col2
        # For the tensor to still be valid, we need the contribution to cancel appropriately
        # This is complex, so let's try a different approach:
        
        # Check if any column is a scalar multiple of another (unlikely in schoolbook)
        # Or try to find if removing a column and adjusting others works
        
        # Alternative: try to find if there's a linear dependency
        # For now, try simple rank reduction by checking if columns are redundant
        
        # Actually, let's try a more systematic approach:
        # Try to find if we can express one column as a combination of others
        # by solving a linear system
        
        if best_rank <= 60:
            break  # Diminishing returns
        
        # Try random linear combination approach
        # Create a new column as random combo of two existing columns
        alpha = np.random.uniform(-2, 2)
        beta = np.random.uniform(-2, 2)
        
        new_U_col = alpha * best_U[:, c1] + beta * best_U[:, c2]
        new_V_col = alpha * best_V[:, c1] + beta * best_V[:, c2]
        new_W_col = alpha * best_W[:, c1] + beta * best_W[:, c2]
        
        # Create candidate with columns c1, c2 replaced by new combination
        # Remove c1 and c2, add new column
        keep_mask = np.ones(best_rank, dtype=bool)
        keep_mask[c1] = False
        keep_mask[c2] = False
        
        cand_U = np.column_stack([best_U[:, keep_mask], new_U_col])
        cand_V = np.column_stack([best_V[:, keep_mask], new_V_col])
        cand_W = np.column_stack([best_W[:, keep_mask], new_W_col])
        
        cand_rank = best_rank - 1
        
        if verify_decomposition(cand_U, cand_V, cand_W, cand_rank):
            best_U = cand_U
            best_V = cand_V
            best_W = cand_W
            best_rank = cand_rank
            print(f"Reduced rank to {best_rank}")
    
    # Convert back to list format with Fraction for exactness
    final_U = np.round(best_U).astype(int)
    final_V = np.round(best_V).astype(int)
    final_W = np.round(best_W).astype(int)
    
    # Final verification
    if not verify_decomposition(final_U.astype(float), final_V.astype(float), final_W.astype(float), best_rank):
        # Fallback to schoolbook
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    U_list = final_U.tolist()
    V_list = final_V.tolist()
    W_list = final_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
