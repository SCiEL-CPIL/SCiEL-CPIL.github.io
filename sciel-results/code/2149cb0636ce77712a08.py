"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import linalg

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p  # 72
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
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p  # 72
    
    # Verify a candidate decomposition
    def verify(U, V, W, rank):
        """Check if U,V,W with given rank exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Search for lower ranks using linear system solving for W
    # Given U and V, the equations for W are linear:
    # sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c] = 1
    # We can solve this system for W.
    
    # Precompute indices
    # Equation index: (i, j, k) -> eq_idx
    # Variable index: (k, i, c) -> var_idx
    
    num_eqs = n * m * p
    # For a given rank r, num_vars = p * n * r
    
    def try_rank(r, num_attempts=200):
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Randomize U and V
            # Use sparse random matrices to increase chance of integer solutions
            # Values in {-2, -1, 0, 1, 2}
            
            U = np.zeros((n * m, r), dtype=float)
            V = np.zeros((m * p, r), dtype=float)
            
            # Fill with random values, keeping some zeros for sparsity
            for i in range(n * m):
                for c in range(r):
                    if np.random.random() < 0.7:
                        U[i, c] = np.random.choice([-2, -1, 1, 2])
            
            for j in range(m * p):
                for c in range(r):
                    if np.random.random() < 0.7:
                        V[j, c] = np.random.choice([-2, -1, 1, 2])
            
            # Build the linear system matrix A and vector b
            # A[eq_idx, var_idx] = U[i,j,c] * V[j,k,c]
            # var_idx corresponds to W[k,i,c]
            # W has shape (p*n, r). Index is (k*n + i) * r + c? 
            # The interface says W[k*n+i, r]. So W is a matrix of size (p*n) x r.
            # So the variable is W[k*n+i, c].
            # Let's flatten W to a vector of length p*n*r.
            # var_idx = (k*n + i) * r + c
            
            num_vars = p * n * r
            A = np.zeros((num_eqs, num_vars), dtype=float)
            b = np.ones(num_eqs, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            u_val = U[i * m + j, c]
                            v_val = V[j * p + k, c]
                            prod = u_val * v_val
                            if prod != 0:
                                var_idx = (k * n + i) * r + c
                                A[eq_idx, var_idx] = prod
                        eq_idx += 1
            
            # Solve A w = b
            # Use least squares if underdetermined
            try:
                w_flat, residuals, rank_A, s = linalg.lstsq(A, b, cond=None)
            except Exception:
                continue
            
            # Check if the solution is close to half-integers
            # w_flat should be close to 0.5 * integer
            # Rescale to integers
            w_int = np.round(w_flat * 2)
            
            # Check error
            error = np.linalg.norm(w_flat * 2 - w_int)
            
            if error < 1e-5:
                # Construct W
                W_flat = w_int / 2.0
                W = W_flat.reshape((p * n, r))
                
                # Verify the solution exactly
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if verify(U_list, V_list, W_list, r):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try to find a rank lower than 72
    # We start from 71 and go down.
    # If we find one, we update best_rank and best_result.
    # We can stop if we find a very low rank, but let's try a few.
    
    # To save time, we can limit the number of attempts per rank.
    # If r is close to 72, it might be easier? Or harder?
    # Actually, the random search is unlikely to succeed for any r < 72 
    # unless there is a specific structure.
    # However, the linear system approach is much more powerful than random W.
    
    # Let's try a few ranks.
    # If we fail to find any, we return schoolbook.
    
    # We can also try to fix U and V to be identity-like or structured.
    # For now, we rely on the random linear solve.
    
    # Optimization: If we find a rank, we can try to reduce it further?
    # No, the search is for a specific rank.
    
    # Let's try ranks 71 down to 65.
    # If time allows, go lower.
    
    for r in range(71, 60, -1):
        if time.time() > deadline - 10:
            break
        
        # Increase attempts for higher ranks (closer to schoolbook) as they might be easier?
        # Or maybe lower ranks are harder.
        # Let's use a fixed number of attempts, say 100.
        result = try_rank(r, num_attempts=100)
        
        if result is not None:
            best_rank = r
            best_result = result
            # If we find a rank, we can try to find an even lower one?
            # The loop continues to lower r.
            # But finding one is already an improvement.
            # We can break if we find a very low rank, but let's see.
            # Actually, if we find rank 71, we should try 70, 69...
            # The loop does this.
            
    return best_result
# EVOLVE-BLOCK-END
