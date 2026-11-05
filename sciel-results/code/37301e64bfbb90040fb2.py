"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Helper to verify a candidate decomposition
    def verify_np(U, V, W, rank):
        """Fast vectorized verification."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Construct the Block Schoolbook Decomposition (Rank 60)
    # Split B (3x8) into B1 (3x3), B2 (3x3), B3 (3x2)
    # Rank 27 + 27 + 6 = 60
    def block_schoolbook():
        rank = 60
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Block 1: k in 0..2
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Block 2: k in 3..5
        for i in range(n):
            for j in range(m):
                for k in range(3, 6):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Block 3: k in 6..7
        for i in range(n):
            for j in range(m):
                for k in range(6, 8):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}

    best_result = block_schoolbook()
    best_rank = 60
    
    # Verify the initial result
    if not verify_np(best_result["U"], best_result["V"], best_result["W"], best_rank):
        # Fallback to full schoolbook if block construction failed (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank

    # Strategy: Try to reduce rank from 60 downwards
    # Use random perturbations and merging
    
    def try_merge_columns(U, V, W, rank, target_rank, max_attempts=10):
        """Try to merge columns to reduce rank."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Randomly select columns to keep and merge others into them
            cols_to_keep = list(range(target_rank))
            cols_to_merge = list(range(target_rank, rank))
            np.random.shuffle(cols_to_merge)
            
            U_new = np.zeros((n * m, target_rank), dtype=float)
            V_new = np.zeros((m * p, target_rank), dtype=float)
            W_new = np.zeros((p * n, target_rank), dtype=float)
            
            # Copy kept columns
            for new_c, old_c in enumerate(cols_to_keep):
                U_new[:, new_c] = U[:, old_c]
                V_new[:, new_c] = V[:, old_c]
                W_new[:, new_c] = W[:, old_c]
            
            # Try to merge remaining columns into kept columns
            for old_c in cols_to_merge:
                # Find the kept column that is "closest" in some metric
                best_match = -1
                best_score = float('inf')
                for new_c in range(target_rank):
                    # Score based on similarity of the triple product contribution
                    # We want to find a new_c such that adding a weighted old_c to new_c
                    # doesn't break the tensor equation.
                    # Approximation: minimize distance between vectors
                    score = np.sum((U[:, old_c] - U_new[:, new_c])**2) + \
                            np.sum((V[:, old_c] - V_new[:, new_c])**2) + \
                            np.sum((W[:, old_c] - W_new[:, new_c])**2)
                    if score < best_score:
                        best_score = score
                        best_match = new_c
                
                if best_match >= 0:
                    # Merge by adding (with small weight to allow cancellation)
                    # We try a few weights
                    weights = [0.5, -0.5, 0.25, -0.25, 1.0, -1.0]
                    merged = False
                    for weight in weights:
                        U_test = U_new.copy()
                        V_test = V_new.copy()
                        W_test = W_new.copy()
                        U_test[:, best_match] += weight * U[:, old_c]
                        V_test[:, best_match] += weight * V[:, old_c]
                        W_test[:, best_match] += weight * W[:, old_c]
                        
                        # Round to half-integers
                        U_test = np.round(U_test * 2) / 2
                        V_test = np.round(V_test * 2) / 2
                        W_test = np.round(W_test * 2) / 2
                        
                        if verify_np(U_test.tolist(), V_test.tolist(), W_test.tolist(), target_rank):
                            return {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
            
            # If no simple merge worked, try random perturbations on the target rank matrices
            # and see if they happen to work (less likely but possible for small ranks)
        
        return None

    # Search for lower ranks
    for target_rank in range(best_rank - 1, max(45, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        # Try multiple attempts
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Start from the current best result (or block schoolbook)
            U = np.array(best_result["U"], dtype=float)
            V = np.array(best_result["V"], dtype=float)
            W = np.array(best_result["W"], dtype=float)
            
            # Add small random perturbations to some entries to help breaking symmetries
            for _ in range(10):
                row_u = np.random.randint(0, n * m)
                col_u = np.random.randint(0, best_rank)
                U[row_u, col_u] += np.random.choice([-0.5, 0.5])
                
                row_v = np.random.randint(0, m * p)
                col_v = np.random.randint(0, best_rank)
                V[row_v, col_v] += np.random.choice([-0.5, 0.5])
                
                row_w = np.random.randint(0, p * n)
                col_w = np.random.randint(0, best_rank)
                W[row_w, col_w] += np.random.choice([-0.5, 0.5])
            
            result = try_merge_columns(U.tolist(), V.tolist(), W.tolist(), best_rank, target_rank)
            if result is not None:
                best_rank = target_rank
                best_result = result
                break
        
        if time.time() > deadline:
            break

    return best_result
# EVOLVE-BLOCK-END
