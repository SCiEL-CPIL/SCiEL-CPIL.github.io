import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 4
    
    # Build the target tensor
    T_expected = np.zeros((9, 12, 12), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(4):
                T_expected[i*3+j, j*4+k, k*3+i] = 1
    
    # Helper to verify a decomposition
    def verify(U, V, W):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.array_equal(T_recon, T_expected)
    
    # Start with the known rank-34 construction
    rank = 34
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((12, rank), dtype=int)
    W = np.zeros((12, rank), dtype=int)
    
    col = 0
    # Strassen Left (A11 * B1L)
    U[0, col] = 1; U[4, col] = 1; V[0, col] = 1; V[5, col] = 1; W[0, col] = 1; W[4, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[0, col] = 1; W[1, col] = 1; W[4, col] = -1; col += 1
    U[0, col] = 1; V[1, col] = 1; V[5, col] = -1; W[3, col] = 1; W[4, col] = 1; col += 1
    U[4, col] = 1; V[4, col] = 1; V[0, col] = -1; W[0, col] = 1; W[1, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[5, col] = 1; W[0, col] = -1; W[3, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[0, col] = 1; V[1, col] = 1; W[4, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[4, col] = 1; V[5, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1
    U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1
    U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1
    U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
    U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
    U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
    U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
    U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
    U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1
    # Strassen Right (A11 * B1R)
    U[0, col] = 1; U[4, col] = 1; V[2, col] = 1; V[7, col] = 1; W[6, col] = 1; W[10, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[2, col] = 1; W[7, col] = 1; W[10, col] = -1; col += 1
    U[0, col] = 1; V[3, col] = 1; V[7, col] = -1; W[9, col] = 1; W[10, col] = 1; col += 1
    U[4, col] = 1; V[6, col] = 1; V[2, col] = -1; W[6, col] = 1; W[7, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[7, col] = 1; W[6, col] = -1; W[9, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[2, col] = 1; V[3, col] = 1; W[10, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[6, col] = 1; V[7, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1
    U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
    U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1
    U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
    U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1
    U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1
    U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1
    U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1
    U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1
    
    # Verify the rank-34 construction
    if not verify(U, V, W):
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce to rank 33 by searching for alternative decompositions
    # Strategy: Use iterative sparsification with linear combination absorption
    # For each pair of columns, try to express one as a linear combination of the other
    # and absorb it into the remaining structure
    
    best_rank = 34
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find rank-33 decomposition by attempting to merge columns
    # Key insight: if column a and column b have proportional U, V, or W vectors,
    # we might be able to merge them
    
    # First, check if any two columns can be merged directly
    # Two columns r1 and r2 can be merged if there exists alpha, beta such that:
    # alpha*U[:,r1] + beta*U[:,r2] = 0 (or similar for V, W)
    # This happens when the column vectors are linearly dependent in a specific way
    
    # More practical approach: try all pairs and see if we can eliminate one
    # by adjusting the other
    
    target_rank = 33
    found_better = False
    
    # Try each column as a candidate for elimination
    for col_to_elim in range(34):
        if time.time() > deadline - 15:
            break
        
        # Create a copy without this column
        mask = [i for i in range(34) if i != col_to_elim]
        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]
        
        # The current reconstruction from remaining columns
        T_remaining = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub, optimize=True)
        T_missing = T_expected - T_remaining
        
        # T_missing should be exactly the contribution of the eliminated column
        # We need to check if we can add this contribution to existing columns
        # by modifying them
        
        # Strategy: try to add the missing tensor to each existing column
        # by solving for adjustments
        
        # For each existing column c, try to find delta_U, delta_V, delta_W such that:
        # (U_sub[:,c] + delta_U) * (V_sub[:,c] + delta_V) * (W_sub[:,c] + delta_W)
        # adds the missing contribution
        
        # This is nonlinear, so use a simpler approach:
        # Check if the missing tensor is already "absorbable" by existing structure
        
        # Alternative: use the fact that if we can find scalars alpha, beta such that
        # the missing contribution equals alpha * (existing column contribution) + 
        # beta * (another existing column contribution), we can merge
        
        # Simpler check: is the missing tensor a scalar multiple of any existing column's contribution?
        col_contributions = []
        for c in range(33):
            contrib = np.einsum('i,j,k->ijk', U_sub[:,c], V_sub[:,c], W_sub[:,c])
            col_contributions.append(contrib)
        
        # Check if T_missing is in the span of existing contributions
        # This is a high-dimensional check, so use a simpler heuristic
        
        # Try to solve: T_missing = sum_c alpha_c * col_contributions[c]
        # This is a linear system if we vectorize the tensors
        
        T_missing_flat = T_missing.flatten()
        
        # Build the coefficient matrix from column contributions
        A_mat = np.column_stack([c.flatten() for c in col_contributions])
        
        # Try to solve A_mat @ alpha = T_missing_flat
        try:
            alpha, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, T_missing_flat, rcond=None)
            
            # Check if the solution is good
            residual = np.linalg.norm(A_mat @ alpha - T_missing_flat)
            
            if residual < 1e-10:
                # We found a valid absorption!
                # Now we need to actually modify the columns
                # This requires solving for the actual U, V, W adjustments
                
                # For simplicity, if alpha has at most 2 non-zero entries, we can merge
                non_zero = np.where(np.abs(alpha) > 1e-10)[0]
                
                if len(non_zero) <= 2:
                    # We can merge into these columns
                    # Create new columns with the absorbed contribution
                    U_new = U_sub.copy()
                    V_new = V_sub.copy()
                    W_new = W_sub.copy()
                    
                    for c in non_zero:
                        if c < 33:
                            U_new[:, c] += alpha[c] * U_sub[:, c]
                            V_new[:, c] += alpha[c] * V_sub[:, c]
                            W_new[:, c] += alpha[c] * W_sub[:, c]
                    
                    # Verify
                    T_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new, optimize=True)
                    if np.array_equal(T_new, T_expected):
                        best_rank = target_rank
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
                        found_better = True
                        break
        except:
            pass
        
        # If direct absorption didn't work, try a different approach:
        # Random perturbation and local search
        if time.time() > deadline - 15:
            break
        
        # Try random linear combinations of columns
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            
            # Pick two random columns to try to merge
            c1 = np.random.randint(0, 33)
            c2 = np.random.randint(0, 33)
            if c1 == c2:
                continue
            
            # Try to express column c1 as a linear combination of c2 and the eliminated column
            # U_sub[:, c1] = alpha * U_sub[:, c2] + beta * U[:, col_to_elim]
            # V_sub[:, c1] = alpha * V_sub[:, c2] + beta * V[:, col_to_elim]
            # W_sub[:, c1] = alpha * W_sub[:, c2] + beta * W[:, col_to_elim]
            
            # This is an overdetermined system, so use least squares
            try:
                # Build the system for U
                AU = np.column_stack([U_sub[:, c1], -U_sub[:, c2], -U[:, col_to_elim]])
                # Actually, we want: U_sub[:, c1] + beta * U[:, col_to_elim] = alpha * U_sub[:, c2]
                # So: alpha * U_sub[:, c2] - beta * U[:, col_to_elim] = U_sub[:, c1]
                
                # For each of U, V, W, we have a system
                # Let's just try to find if there's a simple relationship
                
                # Check if U[:, col_to_elim] is proportional to U_sub[:, c1] - alpha * U_sub[:, c2]
                # for some alpha
                
                # Try alpha = 1
                diff_U = U_sub[:, c1] - U_sub[:, c2]
                if np.allclose(diff_U, 0):
                    # U parts are the same, check V and W
                    diff_V = V_sub[:, c1] - V_sub[:, c2]
                    diff_W = W_sub[:, c1] - W_sub[:, c2]
                    
                    # Check if the differences are proportional to the eliminated column
                    if np.allclose(diff_V, 0) and np.allclose(diff_W, 0):
                        # Perfect merge!
                        best_rank = target_rank
                        mask_final = [i for i in range(34) if i != col_to_elim and i != c1]
                        best_U = U[:, mask_final]
                        best_V = V[:, mask_final]
                        best_W = W[:, mask_final]
                        found_better = True
                        break
            except:
                pass
        
        if found_better:
            break
    
    # If we found a better rank, use it
    if found_better and best_rank < 34:
        U = best_U
        V = best_V
        W = best_W
        rank = best_rank
    else:
        # Stick with rank 34
        U, V, W = U, V, W
        rank = 34
    
    # Final verification
    if not verify(U, V, W):
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
