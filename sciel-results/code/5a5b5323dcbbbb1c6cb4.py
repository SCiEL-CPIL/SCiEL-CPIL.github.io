"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook decomposition for 5x5 matrix multiplication.
    The rank is n*m*p = 125.
    Includes a time guard to ensure safe termination.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0 - 5.0  # 5s safety margin

    # Build schoolbook as safe fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def verify_decomp(U, V, W, rank):
        """Verify exact decomposition with tolerance for float error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def try_decomposition(rank, max_iters=100, seed=0):
        """Try to find a rank decomposition via alternating minimization."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        # Initialize with small integers/halves to help convergence to simple solutions
        # Using a mix of 0, 1, -1 is often better for tensor rank problems
        init_vals = [0, 1, -1]
        U = np.random.choice(init_vals, size=(n*m, rank))
        V = np.random.choice(init_vals, size=(m*p, rank))
        W = np.random.choice(init_vals, size=(p*n, rank))
        
        # Convert to float for lstsq
        U = U.astype(float)
        V = V.astype(float)
        W = W.astype(float)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # B[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    # We want U[i*m+j, :] such that sum_c U[c] * B[c, k] = 1 for all k
                    # This is B @ U[i*m+j, :].T = ones(p)
                    B = np.zeros((rank, p))
                    for c in range(rank):
                        for k in range(p):
                            B[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    # We want V[j*p+k, :] such that sum_c V[c] * A[c, i] = 1 for all i
                    # This is A @ V[j*p+k, :].T = ones(n)
                    A = np.zeros((rank, n))
                    for c in range(rank):
                        for i in range(n):
                            A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # C[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    # We want W[k*n+i, :] such that sum_c W[c] * C[c, j] = 1 for all j
                    # This is C @ W[k*n+i, :].T = ones(m)
                    C = np.zeros((rank, m))
                    for c in range(rank):
                        for j in range(m):
                            C[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence/error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-8:
                # Found a valid decomposition, now try to round to integers/halves
                # First try rounding to integers
                U_int = np.round(U).astype(int)
                V_int = np.round(V).astype(int)
                W_int = np.round(W).astype(int)
                
                err_int = verify_decomp(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank)
                if err_int < 1e-9:
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
                # Try rounding to halves
                U_half = np.round(U * 2) / 2.0
                V_half = np.round(V * 2) / 2.0
                W_half = np.round(W * 2) / 2.0
                
                err_half = verify_decomp(U_half.tolist(), V_half.tolist(), W_half.tolist(), rank)
                if err_half < 1e-9:
                    return U_half.tolist(), V_half.tolist(), W_half.tolist()
                
                # If rounding fails, return the float version (evaluator accepts half-integers, but floats might work if exact enough? No, must be exact)
                # The problem states "integer/half-integer entries". If we can't round, it's not a valid solution for the specific constraint.
                # However, if the float solution is extremely close, maybe we can try a more aggressive rounding or just return the float?
                # The prompt says "exactly decompose". So we must return exact integers or halves.
                # If we can't round, this attempt is a failure for the "exact" requirement, unless we consider the float representation exact in binary?
                # No, 0.1 is not exact. But 0.5 is.
                # Let's assume if we find a float solution with err < 1e-12, it's likely rational with small denominator.
                # But for safety, if we can't round to int/half, we discard this solution.
                pass

        return None

    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find lower rank
    # Known lower bounds for 5x5x5 are higher than 49, but let's search from a reasonable point.
    # Strassen's algorithm for 8x8x8 has rank 7. For 5x5x5, the rank is unknown but likely around 50-60?
    # Actually, the rank of matrix multiplication for n x n is at least n^2. For 5x5, min rank is at least 25.
    # Let's try from 25 up to 124.
    # But 120s limit. We should be careful.
    # Let's try a few specific ranks that are known to be good or close to lower bounds.
    # Or just sweep downwards from 124.
    
    # To save time, we can skip some ranks or use a coarser search first.
    # Let's try ranks: 124, 123, ..., 50.
    # If we find one, we stop.
    
    # Optimization: Use numpy vectorization in the alternating minimization if possible?
    # The inner loops are over i, j, k which are small (5). The rank is the large dimension.
    # The lstsq calls are the bottleneck.
    
    # Let's limit the number of trials per rank to save time.
    trials_per_rank = 2
    
    for rank in range(rank_schoolbook - 1, 24, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        for trial in range(trials_per_rank):
            if time.time() > deadline - 10:
                break
                
            result = try_decomposition(rank, max_iters=50, seed=trial)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            # If we found a lower rank, we can try to go even lower?
            # The loop continues to rank-1 automatically.
            pass

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
# EVOLVE-BLOCK-END
