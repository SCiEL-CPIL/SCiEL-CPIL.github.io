"""Program for matrix multiplication tensor decomposition 4x4 x 4x6."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        """
        Check if U, V, W form a valid tensor decomposition.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_reshaped = U_arr.reshape(n, m, rank)  # (i, j, c)
        V_reshaped = V_arr.reshape(m, p, rank)  # (j, k, c)
        W_reshaped = W_arr.reshape(p, n, rank)  # (k, i, c)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(T, 1.0, atol=1e-10)
    
    def try_rank_reduction(U, V, W, current_rank, target_rank, max_attempts=200):
        """
        Try to reduce rank from current_rank to target_rank by finding
        linear combinations of rank-1 terms that can be merged.
        Uses a greedy approach: try to express some columns as linear
        combinations of others.
        """
        if target_rank >= current_rank:
            return None
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_r = current_rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Try to find a set of columns that are linearly dependent
            # in the U tensor, and see if we can merge them
            # Strategy: pick random pairs and see if one can be expressed
            # as a combination of others
            
            # For simplicity, try removing one column at a time and
            # adjusting the remaining columns to compensate
            col_to_remove = random.randint(0, best_r - 1)
            
            # Check if column col_to_remove is in the span of others
            # for U, V, and W simultaneously
            U_rem = np.delete(best_U, col_to_remove, axis=1)
            V_rem = np.delete(best_V, col_to_remove, axis=1)
            W_rem = np.delete(best_W, col_to_remove, axis=1)
            
            U_col = best_U[:, col_to_remove]
            V_col = best_V[:, col_to_remove]
            W_col = best_W[:, col_to_remove]
            
            # Try to find coefficients such that:
            # U_col ≈ U_rem @ alpha
            # V_col ≈ V_rem @ beta  
            # W_col ≈ W_rem @ gamma
            # But we need the SAME coefficients for all three!
            # This is a joint linear system.
            
            # Stack the constraints:
            # For each row i in U: U_col[i] = sum_j U_rem[i,j] * alpha[j]
            # For each row i in V: V_col[i] = sum_j V_rem[i,j] * beta[j]
            # For each row i in W: W_col[i] = sum_j W_rem[i,j] * gamma[j]
            # We need alpha = beta = gamma = some vector c
            
            # So: U_col = U_rem @ c, V_col = V_rem @ c, W_col = W_rem @ c
            # This is an overdetermined system. Let's solve least squares.
            
            A_stack = np.vstack([U_rem, V_rem, W_rem])  # (nm + mp + pn, best_r-1)
            b_stack = np.concatenate([U_col, V_col, W_col])  # (nm + mp + pn,)
            
            try:
                c, residuals, rank_c, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
            except:
                continue
            
            # Check if the fit is good
            U_pred = U_rem @ c
            V_pred = V_rem @ c
            W_pred = W_rem @ c
            
            if (np.allclose(U_col, U_pred, atol=1e-8) and 
                np.allclose(V_col, V_pred, atol=1e-8) and 
                np.allclose(W_col, W_pred, atol=1e-8)):
                # We can remove this column
                best_U = U_rem
                best_V = V_rem
                best_W = W_rem
                best_r -= 1
                
                if best_r <= target_rank:
                    return {"U": best_U, "V": best_V, "W": best_W}
        
        return None
    
    def strassen_2x2():
        """Strassen decomposition for 2x2 x 2x2 -> 2x2, rank 7."""
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # A: 0=A11, 1=A12, 2=A21, 3=A22
        # B: 0=B11, 1=B12, 2=B21, 3=B22
        # C: 0=C11, 1=C12, 2=C21, 3=C22
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Build schoolbook as baseline
    school = build_schoolbook()
    school_U, school_V, school_W = school["U"], school["V"], school["W"]
    best_rank = n * m * p
    best_result = school
    
    if validate(school_U, school_V, school_W, best_rank):
        best_result = school
    
    random.seed(42)
    
    # Try to reduce rank from schoolbook
    current_U, current_V, current_W = school_U.copy(), school_V.copy(), school_W.copy()
    current_rank = best_rank
    
    # Greedy rank reduction
    while current_rank > 40 and time.time() < deadline:
        result = try_rank_reduction(current_U, current_V, current_W, current_rank, current_rank - 1)
        if result is not None:
            current_U, current_V, current_W = result["U"], result["V"], result["W"]
            current_rank -= 1
            if current_rank < best_rank:
                best_rank = current_rank
                best_result = {"U": current_U, "V": current_V, "W": current_W}
        else:
            # Try a few more random removal attempts
            found = False
            for _ in range(10):
                if time.time() > deadline:
                    break
                col_to_remove = random.randint(0, current_rank - 1)
                U_rem = np.delete(current_U, col_to_remove, axis=1)
                V_rem = np.delete(current_V, col_to_remove, axis=1)
                W_rem = np.delete(current_W, col_to_remove, axis=1)
                
                U_col = current_U[:, col_to_remove]
                V_col = current_V[:, col_to_remove]
                W_col = current_W[:, col_to_remove]
                
                A_stack = np.vstack([U_rem, V_rem, W_rem])
                b_stack = np.concatenate([U_col, V_col, W_col])
                
                try:
                    c, _, _, _ = np.linalg.lstsq(A_stack, b_stack, rcond=None)
                except:
                    continue
                
                U_pred = U_rem @ c
                V_pred = V_rem @ c
                W_pred = W_rem @ c
                
                if (np.allclose(U_col, U_pred, atol=1e-8) and 
                    np.allclose(V_col, V_pred, atol=1e-8) and 
                    np.allclose(W_col, W_pred, atol=1e-8)):
                    current_U, current_V, current_W = U_rem, V_rem, W_rem
                    current_rank -= 1
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_result = {"U": current_U, "V": current_V, "W": current_W}
                    found = True
                    break
            
            if not found:
                break  # Can't reduce further with this approach
    
    # Final validation
    if validate(current_U, current_V, current_W, current_rank):
        if current_rank < best_rank:
            best_rank = current_rank
            best_result = {"U": current_U, "V": current_V, "W": current_W}
    
    return best_result
# EVOLVE-BLOCK-END
