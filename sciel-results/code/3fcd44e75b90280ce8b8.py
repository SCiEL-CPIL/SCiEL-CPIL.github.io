"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _verify_numpy(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition using numpy."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                val = np.dot(u_val, v_val * w_val)
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy broadcasting."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need T[i,j,k] == 1 for all i,j,k
    
    # Reshape and use einsum or manual loops
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _try_rank_r(n, m, p, r, max_attempts=200, seed=42):
    """Try to find a rank-r decomposition using structured random search."""
    rng = np.random.RandomState(seed)
    
    for attempt in range(max_attempts):
        # Initialize with structured sparsity
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Use a pattern where each column has limited support
        # This is inspired by known tensor decomposition structures
        
        for c in range(r):
            # Choose which rows are nonzero in each factor for column c
            # U has n*m rows, V has m*p rows, W has p*n rows
            # For column c, pick a small number of nonzeros
            
            # Strategy: for each column, pick one (i,j) for U, one (j',k) for V, one (k',i') for W
            # But we need consistency: the j in U should match j' in V, and k in V should match k' in W, and i' in W should match i in U
            
            # Simpler: random sparse initialization
            nnz_u = max(1, int(r * 0.3))
            nnz_v = max(1, int(r * 0.3))
            nnz_w = max(1, int(r * 0.3))
            
            # Pick random nonzero positions
            u_rows = rng.choice(n * m, size=min(nnz_u, n*m), replace=False)
            v_rows = rng.choice(m * p, size=min(nnz_v, m*p), replace=False)
            w_rows = rng.choice(p * n, size=min(nnz_w, p*n), replace=False)
            
            for row in u_rows:
                U[row, c] = rng.choice([-1, 1])
            for row in v_rows:
                V[row, c] = rng.choice([-1, 1])
            for row in w_rows:
                W[row, c] = rng.choice([-1, 1])
        
        # Verify
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_structured_pattern(n, m, p, r, seed=0):
    """Try a structured decomposition pattern."""
    rng = np.random.RandomState(seed)
    
    # Pattern: group columns by (i, k) pairs
    # For each (i,k), we need sum_j A[i,j]*B[j,k] to appear
    # A common approach: for each (i,k), use a few terms
    
    # Try: for each i in range(n), k in range(p), assign some columns
    # to compute the dot product over j
    
    # This is essentially trying to find a low-rank factorization
    # of the multiplication tensor
    
    # Alternative: use the fact that for fixed i,k, the contribution is
    # sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need this to equal 1 for each (i,j,k) triple... wait no.
    
    # Actually T[i,j,k] = 1 for ALL i,j,k. So for each triple (i,j,k),
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 100 equations in r unknowns per factor
    # (actually 3*r variables total, but coupled)
    
    # Let's try a different approach: fix U and W, solve for V, or vice versa
    
    # For a given rank r, we have:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try: fix U and W randomly, then for each (j,k), we have:
    # sum_c V[j*p+k,c] * (sum over i of U[i*m+j,c]*W[k*n+i,c]) ... no that's not right
    
    # Actually for fixed j,k: sum_c V[j*p+k,c] * (something depending on i) = 1 for all i
    # This means for fixed j,k, the vector [sum_c V[j*p+k,c]*W[k*n+i,c]]_i must be constant = 1
    # Hmm, this is getting complex.
    
    # Let's just do more random trials with better initialization
    best_result = None
    
    for trial in range(50):
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Dense random initialization with small values
        for c in range(r):
            # Each column: pick random entries
            num_nonzero_u = rng.randint(1, 4)
            num_nonzero_v = rng.randint(1, 4)
            num_nonzero_w = rng.randint(1, 4)
            
            u_idx = rng.choice(n * m, size=num_nonzero_u, replace=False)
            v_idx = rng.choice(m * p, size=num_nonzero_v, replace=False)
            w_idx = rng.choice(p * n, size=num_nonzero_w, replace=False)
            
            for idx in u_idx:
                U[idx, c] = rng.choice([-1.0, 1.0])
            for idx in v_idx:
                V[idx, c] = rng.choice([-1.0, 1.0])
            for idx in w_idx:
                W[idx, c] = rng.choice([-1.0, 1.0])
        
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, n*p) = max(20, 25, 20) = 25
    lower_bound = max(n * m, m * p, n * p)
    
    # Try ranks from lower_bound upward
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple seeds for this rank
        found = None
        for seed in range(20):
            if time.time() > deadline:
                break
            
            result = _try_rank_r(n, m, p, r, max_attempts=100, seed=seed)
            if result is not None:
                found = result
                break
        
        if found is not None:
            best = found
            best_rank = r
            if best_rank <= lower_bound:
                break
    
    return best
# EVOLVE-BLOCK-END
