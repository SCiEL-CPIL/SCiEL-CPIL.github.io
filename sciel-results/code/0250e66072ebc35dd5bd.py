import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s
    
    # Fallback to schoolbook
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Precompute T as a dense array (size: 12 * 32 * 24 = 9216, very small)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute index lists for fast residual computation
    idx_i = list(range(n))
    idx_j = list(range(m))
    idx_k = list(range(p))
    
    def check_exact(U, V, W, tol=1e-8):
        """Check if the decomposition exactly reproduces T."""
        # Compute S = sum_r U[:,r] V[:,r] W[:,r] using einsum
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = np.abs(T - S).max()
        return residual < tol, residual
    
    def round_to_half_integer(M):
        """Round a matrix to nearest half-integers."""
        return np.round(M * 2) / 2.0
    
    def try_round_and_check(U, V, W):
        """Try rounding to half-integers and check if it's still exact."""
        Ur = round_to_half_integer(U)
        Vr = round_to_half_integer(V)
        Wr = round_to_half_integer(W)
        exact, residual = check_exact(Ur, Vr, Wr, tol=1e-6)
        if exact:
            return Ur, Vr, Wr, True
        return None, None, None, False
    
    def als_decompose(rank, U_init, V_init, W_init, max_iters=50, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_residual = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            # design_U shape: (D, rank) where D = m*p*p*n
            # We want to solve design_U @ U[i,:] = T[i,:,:].flatten()
            design_U = np.zeros((m * p * p * n, rank))
            for r in range(rank):
                design_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            U_new = np.zeros((n * m, rank))
            for i in range(n * m):
                target = T[i, :, :].flatten()
                sol, _, _, _ = np.linalg.lstsq(design_U, target, rcond=None)
                U_new[i, :] = sol
            
            U = U_new
            
            # Update V
            design_V = np.zeros((n * m * p * n, rank))
            for r in range(rank):
                design_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            V_new = np.zeros((m * p, rank))
            for j in range(m * p):
                target = T[:, j, :].flatten()
                sol, _, _, _ = np.linalg.lstsq(design_V, target, rcond=None)
                V_new[j, :] = sol
            
            V = V_new
            
            # Update W
            design_W = np.zeros((n * m * m * p, rank))
            for r in range(rank):
                design_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            W_new = np.zeros((p * n, rank))
            for k in range(p * n):
                target = T[:, :, k].flatten()
                sol, _, _, _ = np.linalg.lstsq(design_W, target, rcond=None)
                W_new[k, :] = sol
            
            W = W_new
            
            # Check convergence
            S = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = np.abs(T - S).max()
            
            if abs(prev_residual - residual) < tol:
                break
            prev_residual = residual
        
        return U, V, W, residual
    
    # Try various ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But matrix multiplication tensors often have rank > max dimension
    # Known results: for 3x4x8, let's try ranks from 32 up to 96
    
    ranks_to_try = []
    for r in range(32, min(97, 65)):  # Try up to rank 64 first
        if time.time() > deadline:
            break
        ranks_to_try.append(r)
    
    # Also try some specific ranks
    for r in [40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 92, 96]:
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    
    ranks_to_try = sorted(set(ranks_to_try))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        max_restarts = 15 if rank <= 48 else 8 if rank <= 64 else 5
        
        for seed in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Run ALS
            U, V, W, residual = als_decompose(rank, U, V, W, max_iters=30)
            
            if residual < 1e-8:
                # Found an exact (or near-exact) solution
                # Try to round to half-integers
                Ur, Vr, Wr, exact = try_round_and_check(U, V, W)
                if exact:
                    # Convert to lists of lists
                    U_list = [[float(x) for x in row] for row in Ur]
                    V_list = [[float(x) for x in row] for row in Vr]
                    W_list = [[float(x) for x in row] for row in Wr]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    # Found a solution, but keep searching for lower rank
                    # Actually, we're going from low to high, so first found is best
                    # But let's continue to see if we can find even lower
                    break
                else:
                    # Not exactly half-integer, but close
                    # Use the unrounded solution
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if residual < 1e-6:
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        break
            elif residual < 1e-4 and rank < best_rank:
                # Good approximation, keep as candidate
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if residual < 1e-6:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
    
    # If we found a better rank, return it
    if best_rank < rank_schoolbook:
        return best_result
    else:
        return best_result
# EVOLVE-BLOCK-END
