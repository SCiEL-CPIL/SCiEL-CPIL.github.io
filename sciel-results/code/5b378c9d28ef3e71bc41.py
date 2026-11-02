"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook decomposition
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
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    def verify(U_mat, V_mat, W_mat, r):
        """Verify that U, V, W with rank r correctly decompose matrix multiplication."""
        # Convert to numpy for efficient verification
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        # The decomposition should satisfy:
        # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid triple
        # Actually, the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 for all i,j,k in range(n), range(m), range(p)
        
        # Build the tensor
        T = np.zeros((n, m, p))
        for c in range(r):
            u_col = U_np[:, c]
            v_col = V_np[:, c]
            w_col = W_np[:, c]
            
            # u_col has indices i*m + j
            # v_col has indices j*p + k  
            # w_col has indices k*n + i
            
            # Reshape for broadcasting
            u_reshaped = u_col.reshape(n, m)
            v_reshaped = v_col.reshape(m, p)
            w_reshaped = w_col.reshape(p, n)
            
            # T[i,j,k] += u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_reshaped[i, j] * v_reshaped[j, k] * w_reshaped[k, i]
        
        # Check if T is all ones
        if np.allclose(T, np.ones((n, m, p)), atol=1e-10):
            return True
        return False
    
    # Try to find lower rank decompositions using random search with local optimization
    # This is a heuristic approach - we'll try to find sparse decompositions
    
    def try_rank_reduction(current_U, current_V, current_W, current_rank, max_attempts=100):
        """Try to reduce rank by removing redundant columns."""
        U_np = np.array(current_U, dtype=float)
        V_np = np.array(current_V, dtype=float)
        W_np = np.array(current_W, dtype=float)
        
        # Try removing columns one at a time
        for attempt in range(max_attempts):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Randomly select a column to try removing
            col_to_remove = np.random.randint(0, current_rank)
            
            # Create new matrices without this column
            new_rank = current_rank - 1
            if new_rank <= 0:
                break
            
            U_new = np.delete(U_np, col_to_remove, axis=1)
            V_new = np.delete(V_np, col_to_remove, axis=1)
            W_new = np.delete(W_np, col_to_remove, axis=1)
            
            # Verify
            if verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), new_rank):
                # Try to remove another column
                success, U_final, V_final, W_final, rank_final = try_rank_reduction(
                    U_new.tolist(), V_new.tolist(), W_new.tolist(), new_rank, max_attempts=10
                )
                if success:
                    return True, U_final, V_final, W_final, rank_final
                else:
                    # Keep the first removal
                    return True, U_new.tolist(), V_new.tolist(), W_new.tolist(), new_rank
            else:
                # Try to compensate by adjusting other columns (simple heuristic)
                # For now, just skip this column
                continue
        
        return False, current_U, current_V, current_W, current_rank
    
    # Attempt rank reduction
    success, U_new, V_new, W_new, rank_new = try_rank_reduction(U, V, W, rank, max_attempts=50)
    
    if success and rank_new < best_rank:
        best_rank = rank_new
        best_U = U_new
        best_V = V_new
        best_W = W_new
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
