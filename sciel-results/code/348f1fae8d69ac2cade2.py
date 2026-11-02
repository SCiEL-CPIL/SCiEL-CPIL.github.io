"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_to_quarter_integer(x):
    """Round x to the nearest quarter-integer (for intermediate steps)."""
    return round(x * 4) / 4.0

def _als_step(U, V, W, n, m, p, rank, time_budget=1.0):
    """
    Perform one full alternating minimization step.
    Returns the updated U, V, W and a measure of change.
    """
    start = time.time()
    change = 0.0
    
    # Fix V, W, optimize U
    for i in range(n):
        for j in range(m):
            # A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # We want U[i*m+j, :] such that A_mat @ U[i*m+j, :] = ones(p)
            A_mat = np.zeros((p, rank))
            for k in range(p):
                A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
            b_vec = np.ones(p)
            try:
                new_U_row = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                change += np.sum((new_U_row - U[i*m+j, :])**2)
                U[i*m+j, :] = new_U_row
            except:
                pass
            if time.time() - start > time_budget:
                return U, V, W, change, True # timeout
        
    # Fix U, W, optimize V
    for j in range(m):
        for k in range(p):
            # A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # We want V[j*p+k, :] such that A_mat @ V[j*p+k, :] = ones(n)
            A_mat = np.zeros((n, rank))
            for i in range(n):
                A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
            b_vec = np.ones(n)
            try:
                new_V_row = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                change += np.sum((new_V_row - V[j*p+k, :])**2)
                V[j*p+k, :] = new_V_row
            except:
                pass
            if time.time() - start > time_budget:
                return U, V, W, change, True
        
    # Fix U, V, optimize W
    for i in range(n):
        for k in range(p):
            # A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # We want W[k*n+i, :] such that A_mat @ W[k*n+i, :] = ones(m)
            A_mat = np.zeros((m, rank))
            for j in range(m):
                A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
            b_vec = np.ones(m)
            try:
                new_W_row = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                change += np.sum((new_W_row - W[k*n+i, :])**2)
                W[k*n+i, :] = new_W_row
            except:
                pass
            if time.time() - start > time_budget:
                return U, V, W, change, True

    return U, V, W, change, False

def _try_rank_advanced(rank, n, m, p, seed=42, max_iters=100, time_limit=10.0):
    """
    Try to find a decomposition of the given rank using advanced initialization and ALS.
    Uses Hadamard-inspired structures and multiple restarts.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    # Generate a Hadamard matrix for structured initialization
    # Size must be power of 2 >= max dimension
    max_dim = max(n*m, m*p, p*n, rank)
    H_size = 1
    while H_size < max_dim:
        H_size *= 2
    
    # Simple recursive Hadamard generation
    def gen_hadamard(size):
        if size == 1:
            return np.array([[1.0]])
        H_half = gen_hadamard(size // 2)
        return np.block([[H_half, H_half], [H_half, -H_half]])
    
    H = gen_hadamard(H_size)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with a mix of random and Hadamard structure
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Inject Hadamard structure: take random subsets of H columns
        # This helps break symmetry and provides a better starting point
        if restart % 2 == 0:
            # Scale Hadamard rows
            U_init = H[:n*m, :rank] * 0.1
            V_init = H[n*m:n*m+m*p, :rank] * 0.1
            W_init = H[n*m+m*p:n*m+m*p+p*n, :rank] * 0.1
            # Blend
            U = 0.5 * U + 0.5 * U_init
            V = 0.5 * V + 0.5 * V_init
            W = 0.5 * W + 0.5 * W_init
        
        # Run ALS
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            U, V, W, change, timed_out = _als_step(U, V, W, n, m, p, rank, time_budget=0.5)
            
            # Check convergence
            if change < 1e-6:
                break
            
            # Periodic rounding check
            if it % 10 == 0 and it > 0:
                U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
                V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
                W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
                
                if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                
                # Track best error for potential refinement
                # Compute error: sum of squared residuals
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                            err += (s - 1.0)**2
                if err < best_err:
                    best_err = err
                    best_U = U_rounded.copy()
                    best_V = V_rounded.copy()
                    best_W = W_rounded.copy()
        
        # Final check after ALS loop
        U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
        V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
        W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
        
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    # If we found a good candidate but not exact, try a local polish with finer rounding
    if best_U is not None and best_err < 1.0:
        # Try quarter-integer rounding on the best candidate
        U_q = np.array([[_round_to_quarter_integer(x) for x in row] for row in best_U])
        V_q = np.array([[_round_to_quarter_integer(x) for x in row] for row in best_V])
        W_q = np.array([[_round_to_quarter_integer(x) for x in row] for row in best_W])
        
        # Note: The problem requires half-integers. Quarter-integers are not allowed.
        # So we must stick to half-integers. But we can use the best candidate to
        # re-initialize a short ALS run with a different seed to try to push it over the edge.
        if time.time() - start_time < time_limit - 2:
            U2 = best_U + np.random.randn(n*m, rank) * 0.01
            V2 = best_V + np.random.randn(m*p, rank) * 0.01
            W2 = best_W + np.random.randn(p*n, rank) * 0.01
            for it in range(20):
                U2, V2, W2, _, _ = _als_step(U2, V2, W2, n, m, p, rank, time_budget=0.2)
            U2_r = np.array([[_round_to_half_integer(x) for x in row] for row in U2])
            V2_r = np.array([[_round_to_half_integer(x) for x in row] for row in V2])
            W2_r = np.array([[_round_to_half_integer(x) for x in row] for row in W2])
            if _verify_decomposition(U2_r.tolist(), V2_r.tolist(), W2_r.tolist(), n, m, p):
                return U2_r.tolist(), V2_r.tolist(), W2_r.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Known lower bounds and typical ranks for 4x4x5
    # Schoolbook is 80.
    # Strassen for 4x4 is 49 (if we view it as 2x2 blocks, but dimensions don't align perfectly).
    # For 4x4x4, rank is at most 49 (Strassen) or 48? Actually, 4x4x4 rank is 49.
    # For 4x4x5, it's a bit more complex.
    # Let's search from a reasonable upper bound downwards.
    # 80 is safe. Let's try to find something better.
    # The theoretical lower bound is roughly (4*4*5)^(2/3) * something? No, rank >= n*m, n*p, m*p.
    # rank >= max(16, 20, 20) = 20.
    # But practical bounds are higher.
    
    # Try ranks from 60 down to 40
    # We prioritize finding *any* improvement first, then optimizing.
    # The current program tries 75..55. Let's try a wider range but with strict time limits.
    
    # Strategy:
    # 1. Try to find a decomposition for rank R.
    # 2. If found, update best.
    # 3. Try lower R.
    
    # To maximize score (which is seed/value, so lower rank is better), we want the lowest possible rank.
    # However, finding a low rank decomposition is hard.
    # Let's start with a moderate rank that is likely solvable, then go lower.
    
    # Let's try ranks: 60, 55, 50, 45, 40
    # But we must be careful with time.
    
    # First, let's see if we can beat 80 easily.
    # Try rank 60.
    
    ranks_to_try = [60, 55, 50, 45, 40]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Try multiple seeds for this rank
        for seed in range(5):
            if time.time() > deadline - 20:
                break
                
            # Limit time per rank/seed combination
            time_limit_per_try = min(10.0, deadline - 20 - time.time())
            if time_limit_per_try < 2.0:
                break
                
            result = _try_rank_advanced(rank, n, m, p, seed=seed + 100, max_iters=50, time_limit=time_limit_per_try)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break # Found a valid one for this rank, try lower rank
        
        # If we found a valid one for this rank, we can stop trying this rank with other seeds
        # and move to the next lower rank.
        if best_rank < rank_schoolbook:
            # We found an improvement. Should we keep searching for even lower?
            # Yes, but only if we have time.
            # The loop continues to the next rank.
            pass
    
    # If we still have time, try to polish the best found result?
    # The result is already rounded to half-integers and verified.
    # So no further polish is needed for validity.
    # However, if we found a rank R, maybe we can find R-1?
    # The loop above handles this by trying lower ranks.
    
    return best_result
# EVOLVE-BLOCK-END
