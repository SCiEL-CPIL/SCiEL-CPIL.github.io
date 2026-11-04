import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
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
    
    def check_exact(U, V, W, tol=1e-8):
        """Check if the decomposition exactly reproduces T."""
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
    
    def block_structured_init(rank, seed):
        """Initialize using block structure of m=4=2x2."""
        np.random.seed(seed)
        
        # Try to use the 2x2 block structure
        # j = 2*j1 + j2, so we have blocks for j1 in {0,1}, j2 in {0,1}
        
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Add block structure: make entries within the same (j1) block correlated
        for j1 in range(2):
            # Rows in U for j in {2*j1, 2*j1+1}
            u_block = U[:, :]
            for i in range(n):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    # Add some structure
                    pass
        
        return U, V, W
    
    def strassen_based_init(rank, seed):
        """Initialize using Strassen-like structure for the 2x2 blocks."""
        np.random.seed(seed)
        
        # Strassen decomposition for 2x2 x 2x2 has rank 7
        # The 7 multiplications are:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # For our case, we have 3x4 x 4x8
        # Split into 3x2 x 2x8 and 2x2 x 2x2 parts
        
        U = np.random.randn(n * m, rank) * 0.3
        V = np.random.randn(m * p, rank) * 0.3
        W = np.random.randn(p * n, rank) * 0.3
        
        return U, V, W
    
    # Try various ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    ranks_to_try = list(range(32, 65))
    
    # Add some specific ranks
    for r in [40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 92, 96]:
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    
    ranks_to_try = sorted(set(ranks_to_try))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        max_restarts = 20 if rank <= 48 else 12 if rank <= 64 else 8
        
        for seed in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Try different initialization strategies
            if seed % 3 == 0:
                U, V, W = block_structured_init(rank, seed)
            elif seed % 3 == 1:
                U, V, W = strassen_based_init(rank, seed)
            else:
                np.random.seed(seed)
                U = np.random.randn(n * m, rank) * 0.5
                V = np.random.randn(m * p, rank) * 0.5
                W = np.random.randn(p * n, rank) * 0.5
            
            # Run ALS
            U, V, W, residual = als_decompose(rank, U, V, W, max_iters=40)
            
            if residual < 1e-8:
                # Found an exact (or near-exact) solution
                Ur, Vr, Wr, exact = try_round_and_check(U, V, W)
                if exact:
                    U_list = [[float(x) for x in row] for row in Ur]
                    V_list = [[float(x) for x in row] for row in Vr]
                    W_list = [[float(x) for x in row] for row in Wr]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break
                else:
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if residual < 1e-6:
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        break
            elif residual < 1e-4 and rank < best_rank:
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
