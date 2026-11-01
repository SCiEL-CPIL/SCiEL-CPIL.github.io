"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # Actually, the rank must be at least the max of the "side" dimensions in some sense
    # For 2x3x3: n*m=6, m*p=9, p*n=6
    # A simple lower bound is max(6,9,6) = 9? No, that's not right either.
    # The rank is at least the max of the "flattened" dimensions divided by... 
    # Let's just try from a reasonable range.
    
    # Known: schoolbook is 18. Let's try to beat it.
    # Start from rank 12 and go up.
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, let's try to find a decomposition with rank < 18
    # We'll use a randomized iterative approach
    
    def try_find_decomposition(r, max_iters=5000, seed=42):
        """Try to find a rank-r decomposition using randomized initialization and gradient-free updates."""
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Scale to make the sum of products approximately 1
        # For each (i,j,k), we want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current values
            loss = 0.0
            # For each (i,j,k)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        loss += (val - 1.0) ** 2
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-10:
                return best_U, best_V, best_W, best_loss
            
            # Random perturbation
            # Perturb one random entry in one of the matrices
            mat_idx = rng.randint(0, 3)
            if mat_idx == 0:
                row = rng.randint(0, n*m)
                col = rng.randint(0, r)
                U[row, col] += rng.randn() * 0.1
            elif mat_idx == 1:
                row = rng.randint(0, m*p)
                col = rng.randint(0, r)
                V[row, col] += rng.randn() * 0.1
            else:
                row = rng.randint(0, p*n)
                col = rng.randint(0, r)
                W[row, col] += rng.randn() * 0.1
        
        return best_U, best_V, best_W, best_loss
    
    # Try various ranks
    for r in range(10, 18):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            U, V, W, loss = try_find_decomposition(r, max_iters=2000, seed=seed)
            
            if loss < 1e-8:
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(val - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list of lists with integer/half-integer if possible
                    best_rank = r
                    best_result = (U, V, W)
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Try to clean up to integers/half-integers
        U_list = [[float(U[i, c]) for c in range(best_rank)] for i in range(n*m)]
        V_list = [[float(V[i, c]) for c in range(best_rank)] for i in range(m*p)]
        W_list = [[float(W[i, c]) for c in range(best_rank)] for i in range(p*n)]
        
        # Round to nearest 0.5
        def round_to_half(x):
            return round(x * 2) / 2.0
        
        U_list = [[round_to_half(x) for x in row] for row in U_list]
        V_list = [[round_to_half(x) for x in row] for row in V_list]
        W_list = [[round_to_half(x) for x in row] for row in W_list]
        
        # Verify after rounding
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(best_rank):
                        val += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
