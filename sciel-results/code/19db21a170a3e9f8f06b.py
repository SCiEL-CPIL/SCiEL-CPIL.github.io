import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def _build_rank17():
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify using exact arithmetic with Fractions."""
    rank = U.shape[1]
    Uf = [[Fraction(int(U[i, r]), 1) for r in range(rank)] for i in range(n*m)]
    Vf = [[Fraction(int(V[i, r]), 1) for r in range(rank)] for i in range(m*p)]
    Wf = [[Fraction(int(W[i, r]), 1) for r in range(rank)] for i in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                for r in range(rank):
                    s += Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_merge_terms(U, V, W, n, m, p, idx_a, idx_b):
    """Try to merge terms a and b into a single term.
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    rank = U.shape[1]
    
    # Fixed terms: all except a and b
    fixed_indices = [r for r in range(rank) if r != idx_a and r != idx_b]
    U_fixed = U[:, fixed_indices]
    V_fixed = V[:, fixed_indices]
    W_fixed = W[:, fixed_indices]
    
    # Compute residual: what the merged term must provide
    # residual[i,j,k] = 1 - sum_{r in fixed} U_fixed[i*m+j,r]*V_fixed[j*p+k,r]*W_fixed[k*n+i,r]
    # Also subtract the contributions of terms a and b
    residual = np.ones((n, m, p))
    
    for r in range(len(fixed_indices)):
        Ur = U_fixed[:, r]
        Vr = V_fixed[:, r]
        Wr = W_fixed[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[i,j,k] -= Ur[i*m+j] * Vr[j*p+k] * Wr[k*n+i]
    
    # Subtract term a's contribution
    for i in range(n):
        for j in range(m):
            for k in range(p):
                residual[i,j,k] -= U[idx_a, i*m+j] * V[idx_a, j*p+k] * W[idx_a, k*n+i]
    
    # Subtract term b's contribution
    for i in range(n):
        for j in range(m):
            for k in range(p):
                residual[i,j,k] -= U[idx_b, i*m+j] * V[idx_b, j*p+k] * W[idx_b, k*n+i]
    
    # Now find c, d, e such that c[i*m+j]*d[j*p+k]*e[k*n+i] = residual[i,j,k]
    # Use iterative proportional fitting (IPF) / Sinkhorn-like approach
    # Initialize
    c = np.ones(n*m)
    d = np.ones(m*p)
    e = np.ones(p*n)
    
    # Check if residual has consistent sign pattern
    # For the product to work, residual must be decomposable
    # First check: if residual[i,j,k] = 0 for some (i,j,k), then at least one of c,d,e must be 0 there
    
    # IPF iterations
    converged = False
    for it in range(200):
        # Update c
        for i in range(n):
            for j in range(m):
            # c[i*m+j] should satisfy: c[i*m+j] * sum_k d[j*p+k]*e[k*n+i] = sum_k residual[i,j,k]
                # Actually: for each (i,j,k): c[i*m+j]*d[j*p+k]*e[k*n+i] = residual[i,j,k]
                # So c[i*m+j] = residual[i,j,k] / (d[j*p+k]*e[k*n+i]) for all k where denom != 0
                vals = []
                for k in range(p):
                    denom = d[j*p+k] * e[k*n+i]
                    if abs(denom) > 1e-15:
                        vals.append(residual[i,j,k] / denom)
                if vals:
                    c[i*m+j] = np.mean(vals)
        
        # Update d
        for j in range(m):
            for k in range(p):
                vals = []
                for i in range(n):
                    denom = c[i*m+j] * e[k*n+i]
                    if abs(denom) > 1e-15:
                        vals.append(residual[i,j,k] / denom)
                if vals:
                    d[j*p+k] = np.mean(vals)
        
        # Update e
        for k in range(p):
            for i in range(n):
                vals = []
                for j in range(m):
                    denom = c[i*m+j] * d[j*p+k]
                    if abs(denom) > 1e-15:
                        vals.append(residual[i,j,k] / denom)
                if vals:
                    e[k*n+i] = np.mean(vals)
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred = c[i*m+j] * d[j*p+k] * e[k*n+i]
                    max_err = max(max_err, abs(pred - residual[i,j,k]))
        
        if max_err < 1e-10:
            converged = True
            break
    
    if not converged:
        return None
    
    # Verify the solution
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pred = c[i*m+j] * d[j*p+k] * e[k*n+i]
                if abs(pred - residual[i,j,k]) > 1e-9:
                    return None
    
    # Build new matrices
    new_rank = rank - 1
    U_new = np.zeros((n*m, new_rank))
    V_new = np.zeros((m*p, new_rank))
    W_new = np.zeros((p*n, new_rank))
    
    for new_idx, old_idx in enumerate(fixed_indices):
        U_new[:, new_idx] = U[:, old_idx]
        V_new[:, new_idx] = V[:, old_idx]
        W_new[:, new_idx] = W[:, old_idx]
    
    U_new[:, new_rank-1] = c
    V_new[:, new_rank-1] = d
    W_new[:, new_rank-1] = e
    
    return U_new, V_new, W_new

def _try_merge_with_scaling(U, V, W, n, m, p, idx_a, idx_b, scale_a, scale_b):
    """Try merging with scaled versions of terms a and b."""
    # This is more general: we allow the merged term to be a linear combination
    # For now, just try the basic merge
    return _try_merge_terms(U, V, W, n, m, p, idx_a, idx_b)

def _search_rank16(U17, V17, W17, n, m, p, time_budget=60):
    """Search for rank 16 by trying to merge pairs of terms."""
    start_time = time.time()
    rank = U17.shape[1]
    
    # Try all pairs
    for a in range(rank):
        for b in range(a+1, rank):
            if time.time() - start_time > time_budget:
                return None
            
            result = _try_merge_terms(U17, V17, W17, n, m, p, a, b)
            if result is not None:
                U16, V16, W16 = result
                if _verify(U16, V16, W16, n, m, p):
                    return U16, V16, W16
    
    return None

def _search_rank16_perturbation(U17, V17, W17, n, m, p, time_budget=60):
    """Try to find rank 16 by perturbing the rank-17 solution.
    
    Idea: Take the rank-17 solution, set one term to zero, and see if we can
    adjust the other terms to compensate. This is a local search.
    """
    start_time = time.time()
    rank = U17.shape[1]
    
    # For each term to eliminate, try to adjust others
    for elim in range(rank):
        if time.time() - start_time > time_budget:
            break
        
        # Compute what term 'elim' contributes
        # And what the remaining terms provide
        fixed_indices = [r for r in range(rank) if r != elim]
        
        # The residual that must be filled by adjusting fixed terms
        # is just the contribution of term 'elim'
        # residual[i,j,k] = U17[elim, i*m+j] * V17[elim, j*p+k] * W17[elim, k*n+i]
        
        # Try small perturbations of the fixed terms
        # This is a nonlinear problem, so use a simple approach:
        # For each fixed term, try scaling it
        
        # Actually, let's try a different approach:
        # Use the merge strategy but with more iterations
        
        result = _try_merge_terms(U17, V17, W17, n, m, p, elim, -1)
        # This won't work directly, let's use the pair merge
    
    return None

def _direct_rank16_search(n, m, p, time_budget=30):
    """Direct search for rank-16 solution using structured patterns.
    
    Key insight: For 2x3x3, we can think of it as two 1x3x3 blocks (for i=0 and i=1).
    Each 1x3x3 block has rank at most 3 (since it's just a 3x3 matrix).
    But the coupling between i=0 and i=1 adds complexity.
    
    Try a structured approach:
    - 6 terms for the "diagonal" parts
    - 10 terms for the "off-diagonal" parts
    Total: 16 terms
    """
    start_time = time.time()
    
    # Let me try a specific construction based on the algebraic structure
    # A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Try a construction with 16 terms:
    # Terms 0-5: handle specific (i,j,k) combinations
    # Terms 6-15: handle the rest
    
    U = np.zeros((n*m, 16))
    V = np.zeros((m*p, 16))
    W = np.zeros((p*n, 16))
    
    # This is a guess-based approach, let me try a known pattern
    # Actually, let me use the rank-17 and try to find which term can be absorbed
    
    return None

def _try_remove_single_term(U, V, W, n, m, p):
    """Try removing each single term from the rank-17 solution."""
    rank = U.shape[1]
    for idx in range(rank):
        mask = np.ones(rank, dtype=bool)
        mask[idx] = False
        U2 = U[:, mask]
        V2 = V[:, mask]
        W2 = W[:, mask]
        if _verify(U2, V2, W2, n, m, p):
            return U2, V2, W2
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Get baseline rank 17
    U, V, W = _build_rank17()
    
    # Verify baseline
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 17
    best_U, best_V, best_W = U, V, W
    
    # First: try simply removing one term (fast check)
    if time.time() < deadline - 10:
        result = _try_remove_single_term(U, V, W, n, m, p)
        if result is not None:
            best_rank = 16
            best_U, best_V, best_W = result
    
    # Second: try merging pairs of terms
    if best_rank > 16 and time.time() < deadline - 5:
        time_left = deadline - 5 - time.time()
        result = _search_rank16(U, V, W, n, m, p, time_budget=time_left)
        if result is not None:
            best_rank = 16
            best_U, best_V, best_W = result
    
    # Third: try removing two terms from rank 17
    if best_rank > 15 and time.time() < deadline - 3:
        rank = U.shape[1]
        for idx1 in range(rank):
            if time.time() > deadline - 3:
                break
            for idx2 in range(idx1 + 1, rank):
                mask = np.ones(rank, dtype=bool)
                mask[idx1] = False
                mask[idx2] = False
                U15 = U[:, mask]
                V15 = V[:, mask]
                W15 = W[:, mask]
                if _verify(U15, V15, W15, n, m, p):
                    best_rank = 15
                    best_U, best_V, best_W = U15, V15, W15
                    break
            if best_rank == 15:
                break
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        U, V, W = _build_rank17()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
