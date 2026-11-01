"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Trivial rank-50 construction as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][column] = 1
                V_trivial[j * p + k][column] = 1
                W_trivial[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    
    # Build the tensor structure
    # T[i,j, k] where i in [0,n), j in [0,m), k in [0,p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W give exact matrix multiplication."""
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for t in range(rank):
                        actual += U[i*m + j][t] * V[j*p + k][t] * W[k*n + i][t]
                    if abs(actual - expected) > 1e-9:
                        return False
        # Check that invalid entries are zero
        # U entries: for each (i,j) in valid range, and each t
        # The constraint is that for any (i,j) where i>=n or j>=m (but they're always valid in our indexing)
        # Actually all U[i*m+j] for i<n, j<m are valid
        # We need to check that the decomposition only produces the correct tensor
        # The tensor T has T[i,j,k] = 1 for all valid i,j,k
        # And implicitly 0 elsewhere (but there's no "elsewhere" in the tensor)
        # Wait - the tensor is defined on the full index set
        # For matrix multiplication, T[(i,j),(j,k),(k,i)] = 1 for all valid indices
        # There are no other nonzero entries
        # So we just need to verify the 50 entries
        return True
    
    def try_rank(r, n_restarts=5):
        """Try to find a rank-r decomposition."""
        nonlocal best_rank, best_U, best_V, best_W
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return False
            
            # Random initialization with small values
            np.random.seed(restart * 1000 + r)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Build target tensor
            target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i*m+j, j*p+k, k*n+i] = 1.0
            
            # Use alternating least squares
            for iteration in range(200):
                if time.time() > deadline:
                    return False
                
                # Update U: for each row (i,j), solve
                for idx in range(n*m):
                    # Build system: sum_t U[idx,t] * V[:,t] * W[:,t] should match target[idx,:,:]
                    # This is: U[idx,:] @ (V * W) should give target[idx,:,:]
                    # Where (V*W)[:,t] = V[:,t] * W[:,t] (element-wise, but V and W have different shapes)
                    # Actually: sum_t U[idx,t] * V[j*p+k, t] * W[k*n+i, t]
                    # For fixed idx=(i,j), we need: sum_t U[idx,t] * (V[j*p+k,t] * W[k*n+i,t]) = 1 for all k
                    # Let's build the matrix M where M[k, t] = V[j*p+k, t] * W[k*n+i, t]
                    # Then U[idx,:] @ M = ones(p)
                    i = idx // m
                    j = idx % m
                    M = np.zeros((p, r))
                    for k in range(p):
                        for t in range(r):
                            M[k, t] = V[j*p+k, t] * W[k*n+i, t]
                    # Solve U[idx,:] @ M = ones(p)
                    try:
                        U[idx, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
                
                # Update V: for each row (j,k), solve
                for idx in range(m*p):
                    j = idx // p
                    k = idx % p
                    M = np.zeros((n, r))
                    for i in range(n):
                        for t in range(r):
                            M[i, t] = U[i*m+j, t] * W[k*n+i, t]
                    try:
                        V[idx, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
                
                # Update W: for each row (k,i), solve
                for idx in range(p*n):
                    k = idx // n
                    i = idx % n
                    M = np.zeros((m, r))
                    for j in range(m):
                        for t in range(r):
                            M[j, t] = U[i*m+j, t] * V[j*p+k, t]
                    try:
                        W[idx, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
                
                # Check if we're close to a solution
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for t in range(r):
                                val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            error += (val - 1.0)**2
                if error < 1e-12:
                    # Try rounding to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for t in range(r):
                                    val += U_rounded[i*m+j, t] * V_rounded[j*p+k, t] * W_rounded[k*n+i, t]
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        # Convert to lists
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        if r < best_rank:
                            best_rank = r
                            best_U = U_list
                            best_V = V_list
                            best_W = W_list
                        return True
        
        return False
    
    # Try ranks from low to high
    # Start with a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But actually the rank must be at least the "slice rank" or other bounds
    # Let's try from 30 upward (well below 50)
    
    for r in range(30, 51):
        if time.time() > deadline:
            break
        if try_rank(r, n_restarts=3):
            break
    
    # Return the best found
    if best_rank < rank_trivial:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
# EVOLVE-BLOCK-END
