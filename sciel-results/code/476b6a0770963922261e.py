"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
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


def _verify_exact(U, V, W, n, m, p):
    """Verify exact tensor decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                s = 0
                for c in range(rank):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                if s != 1:
                    return False
    return True


def _compute_residuals(U, V, W, n, m, p):
    """Compute squared residuals for all (i,j,k) positions."""
    rank = len(U[0]) if U else 0
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                s = 0
                for c in range(rank):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                residuals.append((s - 1) ** 2)
    return sum(residuals)


def _local_search_merge(n, m, p, time_limit):
    """
    Local search: start from schoolbook, try to merge pairs of rank-1 terms.
    For each pair (c1, c2), check if we can find a single term c that
    replaces both while keeping all other terms fixed.
    
    A term c is defined by vectors u_c (dim n*m), v_c (dim m*p), w_c (dim p*n).
    The constraint for position (i,j,k) is:
      sum_{c' in active} U[ui,c']*V[vj,c']*W[wk,c'] = 1
    
    If we remove c1 and c2 and add c, we need:
      For all (i,j,k): U[ui,c1]*V[vj,c1]*W[wk,c1] + U[ui,c2]*V[vj,c2]*W[wk,c2]
                        = U[ui,c]*V[vj,c]*W[wk,c]
    
    This is a system of 84 equations in the entries of u_c, v_c, w_c.
    Since each term in schoolbook is a unit vector, the LHS is either 0 or 1
    for each position. We need the new term to match this pattern.
    
    For schoolbook terms, term c1 covers position (i1,j1,k1) and term c2 covers
    (i2,j2,k2). The new term must cover both positions, meaning:
      U[i1*m+j1, c] * V[j1*p+k1, c] * W[k1*n+i1, c] = 1
      U[i2*m+j2, c] * V[j2*p+k2, c] * W[k2*n+i2, c] = 1
      And 0 for all other positions.
    
    This is only possible if the two positions share structure.
    """
    start_time = time.time()
    n, m, p = 3, 4, 7
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    pos_map = {}  # col -> (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                pos_map[col] = (i, j, k)
                col += 1
    
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try all pairs of terms to see if they can be merged
    # Two terms at positions (i1,j1,k1) and (i2,j2,k2) can be merged
    # if there exists a rank-1 tensor that is 1 at both positions and 0 elsewhere.
    # This requires that the two positions are "compatible" in some sense.
    
    # For a rank-1 term to be 1 at (i,j,k) and 0 at all other (i',j',k'),
    # we need: U[i*m+j,c]=a, V[j*p+k,c]=b, W[k*n+i,c]=c with a*b*c=1,
    # and for all other (i',j',k'), U[i'*m+j',c]*V[j'*p+k',c]*W[k'*n+i',c]=0.
    
    # This means at least one of U[i'*m+j',c], V[j'*p+k',c], W[k'*n+i',c] must be 0.
    
    # For two positions (i1,j1,k1) and (i2,j2,k2) to share a rank-1 term,
    # the term must be nonzero at both. Let's say U is nonzero at rows i1*m+j1
    # and i2*m+j2 (if different), V at j1*p+k1 and j2*p+k2, W at k1*n+i1 and k2*n+i2.
    
    # The zero pattern must cover all other positions. This is a constraint satisfaction problem.
    
    # For now, let's try a simpler approach: use a random search with a smooth objective.
    # Start from schoolbook, perturb randomly, and minimize the residual.
    
    import random as _random
    _random.seed(42)
    
    current_rank = n * m * p
    best_residual = 0.0
    best_solution = (U, V, W)
    
    # Try to find a decomposition with rank < current_rank using ALS-like approach
    # with a fixed rank r and optimize the entries.
    
    for target_rank in range(83, 75, -1):
        if time.time() - start_time > time_limit * 0.7:
            break
        # Initialize with random small values
        U_r = [[_random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
        V_r = [[_random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
        W_r = [[_random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
        
        # ALS iterations
        for iteration in range(200):
            if time.time() - start_time > time_limit * 0.8:
                break
            # Update U
            for i in range(n*m):
                for c in range(target_rank):
                    # Gradient of residual w.r.t. U[i,c]
                    grad = 0
                    for j in range(m):
                        for k in range(p):
                            # Position (i//m, j, k) but i is row index in U
                            ii = i // m
                            jj = i % m
                            vj = jj * p + k
                            wk = k * n + ii
                            # d/dU[i,c] of (sum_c' U[i,c']*V[vj,c']*W[wk,c'] - 1)^2
                            # = 2 * (sum_c' ... - 1) * V[vj,c] * W[wk,c]
                            residual_i = 0
                            for c2 in range(target_rank):
                                residual_i += U[i][c2] * V[vj][c2] * W[wk][c2]
                            residual_i -= 1
                            grad += residual_i * V[vj][c] * W[wk][c]
                    # Simple gradient step
                    U[i][c] -= 0.01 * grad
            
            # Update V similarly
            for j in range(m*p):
                for c in range(target_rank):
                    grad = 0
                    jj = j // p
                    kk = j % p
                    for i in range(n):
                        ui = i * m + jj
                        wk = kk * n + i
                        residual_j = 0
                        for c2 in range(target_rank):
                            residual_j += U[ui][c2] * V[j][c2] * W[wk][c2]
                        residual_j -= 1
                        grad += residual_j * U[ui][c] * W[wk][c]
                    V[j][c] -= 0.01 * grad
            
            # Update W similarly
            for k in range(p*n):
                for c in range(target_rank):
                    grad = 0
                    kk = k // n
                    ii = k % n
                    for j in range(m):
                        ui = ii * m + j
                        vj = j * p + kk
                        residual_k = 0
                        for c2 in range(target_rank):
                            residual_k += U[ui][c2] * V[vj][c2] * W[k][c2]
                        residual_k -= 1
                        grad += residual_k * U[ui][c] * V[vj][c]
                    W[k][c] -= 0.01 * grad
        
        # Check if we found a good solution
        res = _compute_residuals(U_r, V_r, W_r, n, m, p)
        if res < 1e-6:
            # Snap to nearest half-integers
            U_snap = [[round(U_r[i][c] * 2) / 2 for c in range(target_rank)] for i in range(n*m)]
            V_snap = [[round(V_r[j][c] * 2) / 2 for c in range(target_rank)] for j in range(m*p)]
            W_snap = [[round(W_r[k][c] * 2) / 2 for c in range(target_rank)] for k in range(p*n)]
            if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
                return {"U": U_snap, "V": V_snap, "W": W_snap}
        elif res < best_residual:
            best_residual = res
            best_solution = (U_r, V_r, W_r)
    
    return None


def _try_lattice_search(n, m, p, max_rank, time_limit):
    """Search for low-rank decomposition."""
    start_time = time.time()
    
    # Try local search with merge strategy
    result = _local_search_merge(n, m, p, time_limit)
    if result:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s kill

    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to improve via search
    try:
        improved = _try_lattice_search(n, m, p, best_rank - 1, deadline - time.time())
        if improved:
            r = len(improved["U"][0])
            if r < best_rank:
                # Verify
                if _verify_exact(improved["U"], improved["V"], improved["W"], n, m, p):
                    best_rank = r
                    result = improved
    except Exception:
        pass

    if time.time() > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
