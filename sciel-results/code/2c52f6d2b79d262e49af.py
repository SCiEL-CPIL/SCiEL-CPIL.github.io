"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    schoolbook_rank = n * m * p  # 75
    
    # Build schoolbook as fallback
    U0 = np.zeros((n * m, schoolbook_rank))
    V0 = np.zeros((m * p, schoolbook_rank))
    W0 = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def is_half_integer(arr, tol=1e-9):
        """Check if all entries are integers or half-integers."""
        for val in arr.flat:
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True
    
    def round_to_half(arr):
        """Round each entry to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def compute_error(U, V, W):
        """Compute sum of squared errors."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1)**2
        return err
    
    # ============================================================
    # STRATEGY: Block decomposition with Strassen for 2x2x2 blocks
    # ============================================================
    # Split A (3x5) into [A1 | A2] where A1 is 3x2, A2 is 3x3
    # Split B (5x5) into [[B11, B12], [B21, B22]] where:
    #   B11: 2x2, B12: 2x3, B21: 3x2, B22: 3x3
    # C = A1*B11 + A2*B21 (3x2) + A1*B12 + A2*B22 (3x3)
    
    # For 2x2x2 blocks, use Strassen (rank 7)
    # Strassen formulas for 2x2 matrices:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B22-B12)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me try a direct approach: enumerate structured decompositions
    # and use optimization to find valid ones.
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But realistically, we need more. Let's try from 45 to 74.
    
    for target_rank in range(45, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 7919 + target_rank * 13 + 42)
            
            # Initialize with structured patterns
            U_curr = np.random.randn(n * m, target_rank) * 0.5
            V_curr = np.random.randn(m * p, target_rank) * 0.5
            W_curr = np.random.randn(p * n, target_rank) * 0.5
            
            # ALS optimization
            converged = False
            max_iterations = 80
            prev_err = float('inf')
            
            for iteration in range(max_iterations):
                if time.time() > deadline:
                    break
                
                # Update U
                U_new = np.zeros_like(U_curr)
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((target_rank, target_rank))
                        for k in range(p):
                            A_mat += np.outer(V_curr[j*p+k, :], W_curr[k*n+i, :])
                        reg = 1e-8 * np.eye(target_rank)
                        try:
                            U_new[i*m+j, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                        except:
                            U_new[i*m+j, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                U_curr = U_new
                
                # Update V
                V_new = np.zeros_like(V_curr)
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((target_rank, target_rank))
                        for i in range(n):
                            A_mat += np.outer(U_curr[i*m+j, :], W_curr[k*n+i, :])
                        reg = 1e-8 * np.eye(target_rank)
                        try:
                            V_new[j*p+k, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                        except:
                            V_new[j*p+k, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                V_curr = V_new
                
                # Update W
                W_new = np.zeros_like(W_curr)
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((target_rank, target_rank))
                        for j in range(m):
                            A_mat += np.outer(U_curr[i*m+j, :], V_curr[j*p+k, :])
                        reg = 1e-8 * np.eye(target_rank)
                        try:
                            W_new[k*n+i, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                        except:
                            W_new[k*n+i, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                W_curr = W_new
                
                err = compute_error(U_curr, V_curr, W_curr)
                
                if err < 1e-14:
                    converged = True
                    break
                if abs(prev_err - err) < 1e-16 and err < 1e-10:
                    converged = True
                    break
                prev_err = err
            
            if converged:
                # Round to half-integers and verify
                U_rounded = round_to_half(U_curr)
                V_rounded = round_to_half(V_curr)
                W_rounded = round_to_half(W_curr)
                
                if verify(U_rounded, V_rounded, W_rounded):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                    break
    
    if best_rank < schoolbook_rank:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U0.tolist()
        V_out = V0.tolist()
        W_out = W0.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
