"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 110  # Safety margin before 120s kill

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1
    U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1
    V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1
    W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1
    W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1
    V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1
    W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1
    V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1
    W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1
    U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1
    W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1
    U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1
    V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1
    U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1
    V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_optimize_rank53(time_budget=50.0):
    """
    Attempt to optimize the rank-53 decomposition to find a rank-52 solution
    or verify the current one. This uses a local search approach on the
    factor matrices to see if we can reduce the effective rank or find
    a valid lower rank decomposition if one exists nearby.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with the known rank-53 solution
    U, V, W = _build_rank53()
    
    # We can't easily reduce rank without a structural change, 
    # but we can try to "compress" the columns if they are linearly dependent 
    # or nearly so, though in exact arithmetic over reals, rank is fixed 
    # unless we find a new decomposition.
    
    # A more promising approach for this specific problem (2x2x15) is 
    # to check if a rank-50 or 51 decomposition exists using the (2,2,3) 
    # block structure.
    # If rank(2,2,3) = 10, then rank(2,2,15) <= 5 * 10 = 50.
    # If rank(2,2,3) = 11, then rank(2,2,15) <= 5 * 11 = 55 (worse than 53).
    # Current best is 53 = 7*7 + 4.
    # 7 blocks of (2,2,2) with rank 7 + 1 block of (2,2,1) with rank 4.
    
    # Let's try to find a rank-10 decomposition for (2,2,3) using random search + least squares.
    # This is a hard problem, but we can try a few random initializations.
    
    def _random_search_223_rank10(seed=42, iterations=100):
        """
        Try to find a rank-10 decomposition for (2,2,3).
        Returns (U, V, W) if found with residual < 1e-6, else None.
        """
        np.random.seed(seed)
        n, m, p = 2, 2, 3
        rank = 10
        
        # Target tensor T[i,j,k] = 1
        # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
        
        # Initialize V and W with random values, then solve for U?
        # Or initialize all and do gradient descent?
        # Given the time constraint, a simple random restart with least squares on U
        # might work if V and W are close to a solution.
        
        # Let's try a structured initialization based on Strassen + Schoolbook
        # and then perturb.
        
        U_str, V_str, W_str = _build_strassen_2x2x2()
        U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
        
        # Build the rank-11 block
        U11 = np.zeros((4, 11))
        V11 = np.zeros((6, 11))
        W11 = np.zeros((6, 11))
        
        U11[:, 0:7] = U_str
        for j in range(2):
            for k in range(2):
                V11[j*3+k, 0:7] = V_str[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W11[k*2+i, 0:7] = W_str[k*2+i, :]
        
        U11[:, 7:11] = U_sb
        for j in range(2):
            V11[j*3+2, 7:11] = V_sb[j, :]
        for i in range(2):
            W11[2*2+i, 7:11] = W_sb[i, :]
        
        # Try to drop one column from the schoolbook part and adjust the rest?
        # Or mix Strassen and Schoolbook terms.
        
        # For now, return None as this is a research-level problem.
        # The strategy in the prompt suggests leveraging linearity in A.
        # Let's try that: Fix V and W to a random basis, solve for U.
        
        best_residual = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for it in range(iterations):
            if time.time() - start_time > time_budget:
                break
            
            # Random initialization for V and W
            V_rand = np.random.randn(6, rank)
            W_rand = np.random.randn(6, rank)
            
            # Construct the linear system for U
            # For each (i, j, k), we have:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is linear in U[i*m+j, r].
            # Let x be the vector of all U entries (size 4*10 = 40).
            # We have 2*2*3 = 12 equations.
            # This is underdetermined (40 unknowns, 12 equations).
            # We want a solution with small norm? Or just any solution?
            # Any solution that satisfies the equations is a valid decomposition.
            # So we just need to find *any* U such that the equations hold.
            # This is always possible if the columns of the coefficient matrix span the target.
            
            # Let's build the coefficient matrix A such that A @ x = b
            # where x is the flattened U.
            
            A = np.zeros((12, 40))
            b = np.ones(12)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation idx: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # The coefficients for U[i*m+j, r] are V[j*p+k, r] * W[k*n+i, r]
                        for r in range(rank):
                            A[idx, (i*m+j)*rank + r] = V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        idx += 1
            
            # Solve A @ x = b
            # Use least squares
            try:
                x, residuals, rank_lu, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check residual
            if residuals and residuals[0] < 1e-6:
                # Found a solution!
                U_sol = x.reshape(4, rank)
                residual = np.linalg.norm(A @ x - b)
                if residual < best_residual:
                    best_residual = residual
                    best_U = U_sol
                    best_V = V_rand
                    best_W = W_rand
                    if residual < 1e-9:
                        break
            
        if best_U is not None:
            return best_U, best_V, best_W
        return None

    # Try a few seeds
    for seed in [42, 123, 999, 314, 271]:
        if time.time() - start_time > time_budget:
            break
        
        result = _random_search_223_rank10(seed=seed, iterations=50)
        if result is not None:
            U223, V223, W223 = result
            # Build full 2x2x15 decomposition
            total_rank = 5 * 10
            U_full = np.zeros((n*m, total_rank))
            V_full = np.zeros((m*p, total_rank))
            W_full = np.zeros((p*n, total_rank))
            
            for b_idx in range(5):
                start_k = b_idx * 3
                col_offset = b_idx * 10
                U_full[:, col_offset:col_offset+10] = U223
                for j in range(m):
                    for k_local in range(3):
                        k_global = start_k + k_local
                        V_full[j*p+k_global, col_offset:col_offset+10] = V223[j*3+k_local, :]
                for k_local in range(3):
                    k_global = start_k + k_local
                    for i in range(n):
                        W_full[k_global*n+i, col_offset:col_offset+10] = W223[k_local*2+i, :]
            
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                return U_full, V_full, W_full, 50
            
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Baseline: rank 53
    U, V, W = _build_rank53()
    best_rank = 53
    best_U, best_V, best_W = U, V, W
    
    # Verify baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = 60
        best_U, best_V, best_W = U, V, W
    
    # Check time
    if time.time() - start_time > TIME_LIMIT - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Attempt to find a better decomposition (e.g., rank 50)
    result = _try_optimize_rank53(time_budget=min(50, TIME_LIMIT - (time.time() - start_time) - 5))
    if result is not None:
        U_new, V_new, W_new, new_rank = result
        if new_rank < best_rank:
            best_rank = new_rank
            best_U, best_V, best_W = U_new, V_new, W_new
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
