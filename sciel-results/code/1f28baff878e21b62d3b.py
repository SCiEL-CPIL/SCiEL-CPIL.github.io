"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 5, 5]
_START_TIME = time.time()
_TIME_LIMIT = 120

# EVOLVE-BLOCK-START
def _verify_decomp(U, V, W, n, m, p, tol=1e-6):
    """Verify exact decomposition: for all i,k,j1,j2: sum_r W[k*n+i,r] U[i*m+j1,r] V[j2*p+k,r] == delta_{j1,j2}"""
    r = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for r_ in range(r):
                        s += W[k*n+i, r_] * U[i*m+j1, r_] * V[j2*p+k, r_]
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > tol:
                        return False
    return True

def _round_half(X):
    """Round matrix to nearest half-integer."""
    return np.round(X * 2.0) / 2.0

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _als_optimize(U, V, W, n, m, p, r, max_iters=300, deadline=None):
    """ALS optimization for the 375-equation system. Returns optimized (U, V, W)."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Update W: for each (k,i), solve linear system
        for k in range(p):
            for i in range(n):
                A_W = np.zeros((m*m, r))
                for j1 in range(m):
                    for j2 in range(m):
                        A_W[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                b = np.zeros(m*m)
                for j in range(m):
                    b[j*m+j] = 1.0
                try:
                    w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                    W[k*n+i, :] = w
                except np.linalg.LinAlgError:
                    pass
        # Update U: for each (i,j1)
        for i in range(n):
            for j1 in range(m):
                A_U = np.zeros((p*m, r))
                b = np.zeros(p*m)
                for k in range(p):
                    for j2 in range(m):
                        A_U[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                        b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                try:
                    u, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                    U[i*m+j1, :] = u
                except np.linalg.LinAlgError:
                    pass
        # Update V: for each (j2,k)
        for j2 in range(m):
            for k in range(p):
                A_V = np.zeros((n*m, r))
                b = np.zeros(n*m)
                for i in range(n):
                    for j1 in range(m):
                        A_V[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                        b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                try:
                    v, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                    V[j2*p+k, :] = v
                except np.linalg.LinAlgError:
                    pass
    return U, V, W

def _try_strassen_2x2(n, m, p):
    """
    Try to exploit Strassen-like structure: decompose the 3x5x5 multiplication
    by treating it as blocks and using Strassen's 7-multiplication formula for 2x2 blocks.
    
    For 3x5x5: think of A as 3x5, B as 5x5.
    We can partition A into rows: row 0 (1x5), rows 1-2 (2x5).
    B into columns: col 0 (5x1), cols 1-2 (5x2), cols 3-4 (5x2).
    
    Actually, let's think more carefully. The standard approach:
    A (3x5) * B (5x5) = C (3x5)
    
    Partition A = [A1; A2] where A1 is 1x5, A2 is 2x5
    Partition B = [B1, B2, B3] where B1 is 5x1, B2 is 5x2, B3 is 5x2
    
    C1 = A1*B1 (1x1) - 5 mults
    C2 = A1*B2 (1x2) - 10 mults
    C3 = A1*B3 (1x2) - 10 mults
    C4 = A2*B1 (2x1) - 10 mults
    C5 = A2*B2 (2x2) - use Strassen: 7 mults instead of 4... wait, 2x2 is 4 mults, Strassen gives 7? No, Strassen is for 2x2*2x2 = 4 mults -> 7 mults? That's worse.
    
    Wait, Strassen reduces 2x2*2x2 from 4 scalar mults to 7? No! Strassen reduces n x n * n x n. For 2x2, standard is 4 mults. Strassen gives 7 mults which is WORSE for 2x2.
    
    Strassen is for when you have larger blocks. The idea is: if you have a 4x4*4x4, standard is 16 mults, Strassen gives 7*(2*2)=28... no.
    
    Actually Strassen: to multiply two n x n matrices, use 7 multiplications of (n/2)x(n/2) matrices. So for 2x2: 7 multiplications of 1x1 = 7 scalar mults vs 4 standard. So Strassen is worse for 2x2.
    
    For 4x4: 7 multiplications of 2x2 = 7*4 = 28 scalar mults vs 16 standard. Still worse.
    
    Strassen only helps asymptotically. For small matrices, it's not helpful.
    
    Let me reconsider the inspiration: "Using Strassen's algorithm for 2x2x2 sub-blocks within a 3x5x5 decomposition reduces the rank from 75 to 71 by exploiting algebraic structure in the 2x2x2 blocks, saving 4 multiplications."
    
    So the idea is different. It's about finding a rank-7 decomposition for a 2x2x2 tensor that shares structure with the 3x5x5 problem.
    
    Actually, I think the key insight is: the 3x5x5 matrix multiplication can be decomposed into independent sub-problems. If we can find that some 2x2x2 sub-blocks can be handled with rank 7 instead of 8, we save multiplications.
    
    Let me think about this differently. The 3x5x5 multiplication has rank at most 75. We want to find a lower rank.
    
    One approach: use the fact that the problem has structure. For example, if we can group some of the 75 terms and show that a subset can be done with fewer terms.
    
    Let me try a different approach: use the ALS optimization from the inspiration, but start from the schoolbook decomposition and try to reduce rank by removing columns and re-optimizing.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT - 10  # safety margin
    
    # Start with schoolbook decomposition
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by trying smaller ranks with ALS optimization
    np.random.seed(42)
    
    # Candidate ranks to try, from lower to higher
    # Lower bound is max(n*m, m*p, n*p) = max(15, 25, 15) = 25
    # But realistically, for 3x5x5, known bounds suggest rank around 60-70
    # Try ranks from 60 up to 74
    candidate_ranks = list(range(60, best_rank))
    
    found_improvement = False
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
            
        # Try multiple random restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_init = np.random.randn(n*m, r) * 0.3
            V_init = np.random.randn(m*p, r) * 0.3
            W_init = np.random.randn(p*n, r) * 0.3
            
            # Run ALS optimization
            U_opt, V_opt, W_opt = _als_optimize(
                U_init.copy(), V_init.copy(), W_init.copy(),
                n, m, p, r, max_iters=100, deadline=deadline - 5
            )
            
            # Round to half-integers and verify
            U_r = _round_half(U_opt)
            V_r = _round_half(V_opt)
            W_r = _round_half(W_opt)
            
            if _verify_decomp(U_r, V_r, W_r, n, m, p):
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    found_improvement = True
                break  # success for this r, try smaller r
            
            # Also try without rounding - maybe the solution is already close to integers
            if _verify_decomp(U_opt, V_opt, W_opt, n, m, p):
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    found_improvement = True
                break
    
    # If no improvement found via ALS, try the Strassen-inspired approach
    # by explicitly constructing a decomposition that shares 2x2x2 sub-blocks
    if not found_improvement and time.time() < deadline - 10:
        # Try to construct a rank-71 decomposition using Strassen-like tricks
        # This is a heuristic: partition the problem and use shared intermediate products
        
        # For now, try a more systematic approach: start from schoolbook and
        # try to merge columns that have similar structure
        
        # Alternative: use the fact that for the 3x5x5 case, we can potentially
        # save rank by exploiting that some of the 75 terms are linearly dependent
        # in a structured way.
        
        # Let's try: take the schoolbook and try to eliminate columns by
        # finding linear dependencies among the tensor slices.
        
        # For each output (k,i), the constraint is:
        # sum_r W[k*n+i,r] U[i*m+j1,r] V[j2*p+k,r] = delta_{j1,j2}
        
        # This is a system of 375 equations. The schoolbook has 75 free parameters
        # per... actually, the schoolbook has exactly 75 rank-1 terms.
        
        # Let's try a different approach: use the ALS but start from a perturbed
        # schoolbook with some columns removed.
        
        for r in range(70, best_rank):
            if time.time() > deadline - 5:
                break
            if r >= best_rank:
                break
            
            # Start from schoolbook, remove (best_rank - r) random columns, then optimize
            U0, V0, W0 = _build_schoolbook(n, m, p)
            
            # Remove random columns
            cols_to_remove = np.random.choice(best_rank, best_rank - r, replace=False)
            cols_to_keep = np.setdiff1d(np.arange(best_rank), cols_to_remove)
            
            U_init = U0[:, cols_to_keep].copy()
            V_init = V0[:, cols_to_keep].copy()
            W_init = W0[:, cols_to_keep].copy()
            
            # Add small random perturbation to break symmetry
            U_init += np.random.randn(n*m, r) * 0.1
            V_init += np.random.randn(m*p, r) * 0.1
            W_init += np.random.randn(p*n, r) * 0.1
            
            U_opt, V_opt, W_opt = _als_optimize(
                U_init, V_init, W_init,
                n, m, p, r, max_iters=200, deadline=deadline - 5
            )
            
            U_r = _round_half(U_opt)
            V_r = _round_half(V_opt)
            W_r = _round_half(W_opt)
            
            if _verify_decomp(U_r, V_r, W_r, n, m, p):
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                break
    
    # Final check: if we still have the schoolbook, return it
    # Convert to lists for the interface
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
