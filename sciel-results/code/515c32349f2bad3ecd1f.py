"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Trivial rank: n*m*p = 105
    trivial_rank = n * m * p
    
    # Build the trivial decomposition
    U = np.zeros((n * m, trivial_rank), dtype=float)
    V = np.zeros((m * p, trivial_rank), dtype=float)
    W = np.zeros((p * n, trivial_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Precompute valid triples (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    # Also build lookup structures for fast constraint extraction
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    
    # Build per-row constraint lists for fast ALS updates
    # For each a: list of (b, c) pairs
    a_to_bc = {a: [] for a in range(n * m)}
    # For each b: list of (a, c) pairs
    b_to_ac = {b: [] for b in range(m * p)}
    # For each c: list of (a, b) pairs
    c_to_ab = {c: [] for c in range(p * n)}
    
    for (a, b, c) in triples:
        a_to_bc[a].append((b, c))
        b_to_ac[b].append((a, c))
        c_to_ab[c].append((a, b))
    
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-8):
        """Vectorized check: for each (a,b,c) triple, sum_t U[a,t]*V[b,t]*W[c,t] == 1"""
        rank = U_mat.shape[1]
        # Compute all needed products efficiently
        # For each triple (a,b,c): need sum_t U[a,t]*V[b,t]*W[c,t]
        # This is a 3-tensor contraction along the rank axis
        # Precompute UV and VW? No, just do it directly for speed with small dims
        # Use numpy broadcasting: for each triple, compute element-wise product and sum
        a_idx = np.array([t[0] for t in triples])
        b_idx = np.array([t[1] for t in triples])
        c_idx = np.array([t[2] for t in triples])
        
        # U[a,t] for all triples: shape (len(triples), rank)
        U_vals = U_mat[a_idx, :]  # (105, rank)
        V_vals = V_mat[b_idx, :]  # (105, rank)
        W_vals = W_mat[c_idx, :]  # (105, rank)
        
        # Element-wise product and sum along rank axis
        products = U_vals * V_vals * W_vals  # (105, rank)
        sums = products.sum(axis=1)  # (105,)
        
        if np.any(np.abs(sums - 1.0) > tol):
            return False
        return True
    
    def time_left():
        return deadline - time.time()
    
    def try_rank(r, max_iters=150, n_seeds=4):
        """Try to find a rank-r decomposition using ALS with precomputed constraints."""
        if r >= trivial_rank:
            return None
        
        for seed in range(n_seeds):
            if time_left() < 8:
                break
            
            np.random.seed(seed * 1000 + 42)
            
            # Random initialization with Toeplitz-like structure hint
            U_mat = np.random.randn(n * m, r) * 0.3
            V_mat = np.random.randn(m * p, r) * 0.3
            W_mat = np.random.randn(p * n, r) * 0.3
            
            for it in range(max_iters):
                if time_left() < 5:
                    break
                
                # Update U: for each row a, solve linear system
                for a in range(n * m):
                    bc_list = a_to_bc[a]
                    if not bc_list:
                        continue
                    # Build constraint matrix: rows are V[b]*W[c] for each (b,c)
                    rows = np.array([V_mat[b] * W_mat[c] for b, c in bc_list])
                    rhs = np.ones(len(bc_list))
                    if rows.shape[0] >= r:
                        x, _, _, _ = np.linalg.lstsq(rows, rhs, rcond=None)
                        U_mat[a, :] = x
                
                # Update V: for each row b, solve linear system
                for b in range(m * p):
                    ac_list = b_to_ac[b]
                    if not ac_list:
                        continue
                    rows = np.array([U_mat[a] * W_mat[c] for a, c in ac_list])
                    rhs = np.ones(len(ac_list))
                    if rows.shape[0] >= r:
                        x, _, _, _ = np.linalg.lstsq(rows, rhs, rcond=None)
                        V_mat[b, :] = x
                
                # Update W: for each row c, solve linear system
                for c in range(p * n):
                    ab_list = c_to_ab[c]
                    if not ab_list:
                        continue
                    rows = np.array([U_mat[a] * V_mat[b] for a, b in ab_list])
                    rhs = np.ones(len(ab_list))
                    if rows.shape[0] >= r:
                        x, _, _, _ = np.linalg.lstsq(rows, rhs, rcond=None)
                        W_mat[c, :] = x
                
                # Check convergence periodically
                if it % 10 == 9:
                    if verify_fast(U_mat, V_mat, W_mat, tol=1e-6):
                        # Try to clean up to get integer/half-integer entries
                        U_clean = np.round(U_mat * 2) / 2
                        V_clean = np.round(V_mat * 2) / 2
                        W_clean = np.round(W_mat * 2) / 2
                        if verify_fast(U_clean, V_clean, W_clean, tol=1e-6):
                            return (U_clean, V_clean, W_clean)
                        else:
                            # Try rounding to integers
                            U_int = np.round(U_mat)
                            V_int = np.round(V_mat)
                            W_int = np.round(W_mat)
                            if verify_fast(U_int, V_int, W_int, tol=1e-6):
                                return (U_int, V_int, W_int)
                            else:
                                return (U_mat, V_mat, W_mat)
        
        return None
    
    # Try to find a lower rank decomposition using sliding window / Toeplitz embedding
    # Key insight: the 5-dim inner product means that for fixed i,k, varying j
    # gives overlapping support. We can try structured initializations.
    
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n * m, m * p, p * n)
    
    # Fallback: trivial decomposition
    U_best = U.copy()
    V_best = V.copy()
    W_best = W.copy()
    best_rank = trivial_rank
    
    # Strategy 1: Try structured "sliding window" initialization
    # For the 5-dim inner product, think of it as a convolution.
    # A rank-one term (u, v, w) where u has support on {j, j+1}, v on {j, j+1}
    # covers multiple (i,j,k) entries.
    # 
    # Actually, let's try a different approach: use the fact that the tensor
    # has a Toeplitz structure along the j dimension.
    # 
    # For now, try ALS with various ranks starting from lower_bound.
    
    for r in range(lower_bound, trivial_rank + 1):
        if time_left() < 12:
            break
        
        result = try_rank(r, max_iters=80, n_seeds=3)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break
    
    # If we still have time, try to reduce rank further from best_rank
    if best_rank > lower_bound and time_left() > 20:
        for r in range(max(lower_bound, best_rank - 5), best_rank):
            if time_left() < 10:
                break
            result = try_rank(r, max_iters=60, n_seeds=2)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = r
                break
    
    # Convert to lists for the required interface
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
