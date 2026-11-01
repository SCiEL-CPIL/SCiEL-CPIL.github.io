"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Combined approach:
1. Structural Rank 18 solution (Strassen + Schoolbook) as baseline.
2. Search for lower ranks (17, 16) using ALS with structured initializations.
3. Randomized algebraic elimination over rationals with sparsity constraints.
4. Verify exactness using Fractions.
"""

import numpy as np
import time
import itertools
from fractions import Fraction
from collections import defaultdict

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T


def _build_target_float():
    return _build_target().astype(np.float64)


def _check_decomposition_float(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    R = U.shape[1]
    result = np.zeros((T.shape[0], T.shape[1], T.shape[2]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=1e-9)


def _compute_tensor(U, V, W):
    """Compute the tensor from factor matrices."""
    R = U.shape[1]
    n, m, p = DIMENSIONS
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _snap_to_half_integers(M):
    """Snap matrix values to nearest half-integer."""
    return np.round(M * 2) / 2.0


def _snap_to_integers(M):
    """Snap matrix values to nearest integer."""
    return np.round(M)

def _to_fraction(val):
    """Convert a float to Fraction, preferring small denominators."""
    if abs(val) < 1e-12:
        return Fraction(0)
    # Try half-integer first
    nearest_half = np.round(val * 2) / 2.0
    if abs(val - nearest_half) < 1e-9:
        return Fraction(int(nearest_half * 2), 2)
    # Try integer
    nearest_int = np.round(val)
    if abs(val - nearest_int) < 1e-9:
        return Fraction(int(nearest_int))
    # Try other small denominators
    for denom in [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100]:
        num = np.round(val * denom)
        if abs(val - num / denom) < 1e-9:
            frac = Fraction(int(num), denom)
            return frac
    # Last resort
    return Fraction(val).limit_denominator(200)


def _check_decomposition_exact(U, V, W, T):
    """Exact check using Fraction to avoid floating point issues."""
    n, m, p = DIMENSIONS
    U_frac = np.zeros_like(U, dtype=object)
    V_frac = np.zeros_like(V, dtype=object)
    W_frac = np.zeros_like(W, dtype=object)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U_frac[i, j] = _to_fraction(U[i, j])
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V_frac[i, j] = _to_fraction(V[i, j])
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_frac[i, j] = _to_fraction(W[i, j])
    
    R = U.shape[1]
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[u][r] * V_frac[v][r] * W_frac[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _get_rank18_solution():
    """
    Returns the structural Rank 18 solution:
    Two Strassen blocks (cols 0-1, 2-3) and Schoolbook for col 4.
    """
    n, m, p = DIMENSIONS
    rank = 18
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    def add_strassen_block(col0, col1, base):
        # M1: (a00+a11)(b00+b11)
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        # M2: (a10+a11)(b00)
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        # M3: (a00)(b11-b01)
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        # M4: (a11)(b10-b00)
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        # M5: (a00+a01)(b11)
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        # M6: (a10-a00)(b00+b01)
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        # M7: (a01-a11)(b10+b11)
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for col 4
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W


def _als_update_U(T, V, W, R):
    n, m, p = DIMENSIONS
    U_rows = n * m
    VW = np.einsum('vr,wr->vwr', V, W)
    VW_flat = VW.reshape(-1, R)
    U = np.zeros((U_rows, R), dtype=np.float64)
    for u in range(U_rows):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(VW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_update_V(T, U, W, R):
    n, m, p = DIMENSIONS
    V_rows = m * p
    UW = np.einsum('ur,wr->uwr', U, W)
    UW_flat = UW.reshape(-1, R)
    V = np.zeros((V_rows, R), dtype=np.float64)
    for v in range(V_rows):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(UW_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_update_W(T, U, V, R):
    n, m, p = DIMENSIONS
    W_rows = p * n
    UV = np.einsum('ur,vr->uvr', U, V)
    UV_flat = UV.reshape(-1, R)
    W = np.zeros((W_rows, R), dtype=np.float64)
    for w in range(W_rows):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _compute_error(U, V, W, T):
    result = _compute_tensor(U, V, W)
    return np.linalg.norm(result - T)

def _als_polish(U, V, W, T, R, max_iters=100, tol=1e-15):
    """High-precision ALS polish."""
    prev_err = np.inf
    for it in range(max_iters):
        U = _als_update_U(T, V, W, R)
        V = _als_update_V(T, U, W, R)
        W = _als_update_W(T, U, V, R)
        err = _compute_error(U, V, W, T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _try_als_rank(R, T, deadline, n_seeds=50, max_iters=50):
    """
    Try to find a rank-R decomposition using ALS with multiple initialization strategies.
    """
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    T_float = T.astype(np.float64)
    
    # Use Rank 18 solution as a structured initialization
    U18, V18, W18 = _get_rank18_solution()
    
    best_err = np.inf
    best_solution = None
    
    seeds = [42, 123, 7, 999, 2024, 555, 777, 1111, 2222, 3333, 
             1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1111, 2222, 3333, 4444, 5555]
    
    for seed_idx in range(min(n_seeds, len(seeds))):
        if time.time() > deadline:
            break
        seed = seeds[seed_idx]
        np.random.seed(seed)
        
        # Multiple initialization strategies
        inits = []
        
        # Strategy 1: Pure random
        U = np.random.randn(U_rows, R) * 0.5
        V = np.random.randn(V_rows, R) * 0.5
        W = np.random.randn(W_rows, R) * 0.5
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # Strategy 2: Random with different scales
        U = np.random.randn(U_rows, R)
        V = np.random.randn(V_rows, R)
        W = np.random.randn(W_rows, R)
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # Strategy 3: Sparse random
        U = np.random.randn(U_rows, R) * (np.random.rand(U_rows, R) < 0.3)
        V = np.random.randn(V_rows, R) * (np.random.rand(V_rows, R) < 0.3)
        W = np.random.randn(W_rows, R) * (np.random.rand(W_rows, R) < 0.3)
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # Strategy 4: For R=17, try structured init from rank-18
        if R == 17:
            # Take first 17 columns of rank-18 and add small perturbation
            U_s = U18[:, :17].copy() + np.random.randn(U_rows, 17) * 0.01
            V_s = V18[:, :17].copy() + np.random.randn(V_rows, 17) * 0.01
            W_s = W18[:, :17].copy() + np.random.randn(W_rows, 17) * 0.01
            inits.append((U_s, V_s, W_s))
            
            # Try dropping different columns from rank-18
            for drop_col in range(18):
                mask = [i for i in range(18) if i != drop_col]
                U_s = U18[:, mask].copy() + np.random.randn(U_rows, 17) * 0.01
                V_s = V18[:, mask].copy() + np.random.randn(V_rows, 17) * 0.01
                W_s = W18[:, mask].copy() + np.random.randn(W_rows, 17) * 0.01
                inits.append((U_s, V_s, W_s))
        
        for U_init, V_init, W_init in inits:
            if time.time() > deadline:
                break
            
            U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                U = _als_update_U(T_float, V, W, R)
                V = _als_update_V(T_float, U, W, R)
                W = _als_update_W(T_float, U, V, R)
                
                err = _compute_error(U, V, W, T_float)
                
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
            
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            # Check for exact solution
            if _check_decomposition_float(U, V, W, T_float):
                # Polish with high precision
                U, V, W, err = _als_polish(U, V, W, T_float, R, max_iters=50, tol=1e-16)
                
                # Try to rationalize
                for snap_fn in [_snap_to_half_integers, _snap_to_integers]:
                    U_s = snap_fn(U)
                    V_s = snap_fn(V)
                    W_s = snap_fn(W)
                    
                    if _check_decomposition_exact(U_s, V_s, W_s, T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                
                # Check if original is exact
                if _check_decomposition_exact(U, V, W, T):
                    return U.tolist(), V.tolist(), W.tolist()
    
    # If we found a very low error solution, try one more polish
    if best_solution is not None and best_err < 1e-6:
        U, V, W = best_solution
        U, V, W, err = _als_polish(U, V, W, T_float, R, max_iters=200, tol=1e-16)
        if _check_decomposition_float(U, V, W, T_float):
            for snap_fn in [_snap_to_half_integers, _snap_to_integers]:
                U_s = snap_fn(U)
                V_s = snap_fn(V)
                W_s = snap_fn(W)
                if _check_decomposition_exact(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            if _check_decomposition_exact(U, V, W, T):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _try_algebraic_elimination_rank17(deadline, n_seeds=50):
    """
    Randomized algebraic elimination strategy for rank-17.
    
    Treats the rank-17 decomposition as a sparse linear system.
    Uses Gaussian elimination over rationals to solve for factor entries
    while enforcing sparsity constraints that break block-diagonal symmetry.
    
    Key insight: For a rank-R decomposition of T with entries in {0,1},
    we have sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w].
    
    We can fix some entries (sparsity pattern) and solve for others.
    The strategy:
    1. Choose a random sparse pattern for U, V, W with exactly R non-zero per row/col
       (or similar structured sparsity)
    2. Fix most entries to small integers/half-integers
    3. Solve the remaining linear system over rationals
    4. Verify the solution
    """
    n, m, p = DIMENSIONS
    R = 17
    U_rows = n * m  # 4
    V_rows = m * p  # 10
    W_rows = p * n  # 10
    
    # Build target tensor
    T = _build_target()
    
    # List of all (u,v,w) positions where T[u,v,w] = 1
    target_positions = []
    for u in range(U_rows):
        for v in range(V_rows):
            for w in range(W_rows):
                if T[u, v, w] == 1:
                    target_positions.append((u, v, w))
    
    # Each equation: sum_r U[u,r]*V[v,r]*W[w,r] = 1 for target positions
    # We have U_rows * V_rows * W_rows = 4*10*10 = 400 potential equations
    # But only 20 are non-zero (the ones in target_positions)
    
    # Strategy: Fix a sparse pattern and solve
    # For each rank term r, we have U[:,r], V[:,r], W[:,r]
    # Total unknowns: 4*17 + 10*17 + 10*17 = 68 + 170 + 170 = 408
    # Equations: 20 (from target positions) + we can add "zero" equations
    # Actually, we need: sum_r U[u,r]*V[v,r]*W[w,r] = 0 for non-target positions too
    # That's 400 - 20 = 380 more equations
    # Total: 400 equations, 408 unknowns - underdetermined
    
    # Better approach: Use the structure more cleverly
    # Fix entries to be in {-1, 0, 1, 1/2} and search
    
    # Alternative: Use the fact that for matrix multiplication,
    # we can think of this as: for each output element (i,k),
    # (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So for 2x2 * 2x5, we have 10 output elements, each needing
    # a sum of products.
    
    # Let's try a different approach: enumerate possible sparse patterns
    # and use rational Gaussian elimination
    
    # For rank 17, we need to "save" 1 rank compared to rank 18.
    # Rank 18 = 2 Strassen blocks (7+7) + 4 schoolbook = 18
    # To get rank 17, we need to share 1 term between blocks or
    # find a clever cross-term cancellation.
    
    # Let's try: fix the Strassen structure for cols 0-3 (14 terms)
    # and try to do col 4 with only 3 terms instead of 4.
    # This requires rank 3 for a 2x2*2x1 multiplication.
    # But rank(2x2x1) = 4, so this won't work directly.
    
    # Instead, try mixing: use some cross-terms between the blocks.
    
    # New strategy: Parameterize with cross-block terms
    # Start from rank-18 and try to merge two terms into one with
    # appropriate coefficients.
    
    U18, V18, W18 = _get_rank18_solution()
    
    # Try to find a rank-17 by searching over possible "merges"
    # of two rank-18 terms into one, with rational coefficients.
    
    # For each pair (r1, r2) from the 18 terms, try to express
    # U[:,r1]*V[:,r1]*W[:,r1] + U[:,r2]*V[:,r2]*W[:,r2]
    # as a single rank-1 term with rational coefficients.
    
    # This is equivalent to finding alpha, beta such that:
    # alpha*U1*V1*W1 + beta*U2*V2*W2 = U3*V3*W3
    # for some U3, V3, W3.
    
    # More practically: try to find a 3-term replacement for 2 terms
    # that has the same effect but uses a "shared" structure.
    
    # Let's try a more systematic approach:
    # Fix U and V to be sparse with entries in {-1, 0, 1, 1/2}
    # and solve for W.
    
    best_result = None
    
    seeds = list(range(200))
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Generate sparse U with entries in {-1, 0, 1, 1/2}
        U_vals = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(U_rows, R))
        V_vals = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(V_rows, R))
        
        # Ensure each column has at least one non-zero in U and V
        for r in range(R):
            if np.all(U_vals[:, r] == 0):
                U_vals[np.random.randint(U_rows), r] = np.random.choice([-1, 1])
            if np.all(V_vals[:, r] == 0):
                V_vals[np.random.randint(V_rows), r] = np.random.choice([-1, 1])
        
        # Build the linear system for W
        # For each (u,v,w): sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w]
        # This is linear in W if we fix U and V.
        # For fixed (u,v): sum_r (U[u,r]*V[v,r])*W[w,r] = T[u,v,w]
        # This gives us, for each (u,v), a linear equation in W[:,r] for all w.
        
        # Actually, for each (u,v), we have W_rows equations:
        # sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w] for w=0..W_rows-1
        # This is: (U[u,:] * V[v,:]) dot W[w,:] = T[u,v,w]
        # Let c_r = U[u,r]*V[v,r], then sum_r c_r * W[w,r] = T[u,v,w]
        # This is W[w,:] dot c = T[u,v,w]
        
        # For all (u,v) pairs, we get U_rows*V_rows = 40 linear equations
        # in W (which has W_rows*R = 170 unknowns)
        # Under-determined, so we need to add sparsity constraints.
        
        # Instead, let's fix W to be sparse too and check consistency.
        W_vals = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(W_rows, R))
        for r in range(R):
            if np.all(W_vals[:, r] == 0):
                W_vals[np.random.randint(W_rows), r] = np.random.choice([-1, 1])
        
        # Check if this decomposition works
        if _check_decomposition_float(U_vals, V_vals, W_vals, T.astype(np.float64)):
            # Try to rationalize and verify exactly
            U_frac = np.zeros((U_rows, R), dtype=object)
            V_frac = np.zeros((V_rows, R), dtype=object)
            W_frac = np.zeros((W_rows, R), dtype=object)
            
            for i in range(U_rows):
                for j in range(R):
                    U_frac[i, j] = _to_fraction(U_vals[i, j])
            for i in range(V_rows):
                for j in range(R):
                    V_frac[i, j] = _to_fraction(V_vals[i, j])
            for i in range(W_rows):
                for j in range(R):
                    W_frac[i, j] = _to_fraction(W_vals[i, j])
            
            # Verify exactly
            valid = True
            for u in range(U_rows):
                for v in range(V_rows):
                    for w in range(W_rows):
                        total = Fraction(0)
                        for r in range(R):
                            total += U_frac[u, r] * V_frac[v, r] * W_frac[w, r]
                        if total != Fraction(int(T[u, v, w])):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_result = (U_vals.tolist(), V_vals.tolist(), W_vals.tolist())
                break
    
    # Try a more structured approach: modify the rank-18 solution
    # by introducing cross-block terms
    
    if best_result is None and time.time() < deadline:
        # Try to find a rank-17 by perturbing the rank-18 solution
        # and using ALS with rational snapping
        
        T_float = T.astype(np.float64)
        
        for seed in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + seed)
            
            # Start from rank-18 and try to "compress" by making
            # two terms nearly identical, then merging.
            
            # Pick two terms to potentially merge
            r1, r2 = np.random.choice(18, 2, replace=False)
            
            # Create a rank-17 initialization by averaging these two terms
            U_init = U18[:, :17].copy()
            V_init = V18[:, :17].copy()
            W_init = W18[:, :17].copy()
            
            # Replace term 16 with a combination of terms r1 and r2
            # This is heuristic - we're trying to find a direction
            # in which the tensor can be expressed with fewer terms.
            
            # Add small random perturbation to escape local minima
            U_init += np.random.randn(U_rows, 17) * 0.05
            V_init += np.random.randn(V_rows, 17) * 0.05
            W_init += np.random.randn(W_rows, 17) * 0.05
            
            # Run ALS
            U_cur, V_cur, W_cur = U_init, V_init, W_init
            prev_err = np.inf
            
            for it in range(100):
                if time.time() > deadline:
                    break
                
                U_cur = _als_update_U(T_float, V_cur, W_cur, R)
                V_cur = _als_update_V(T_float, U_cur, W_cur, R)
                W_cur = _als_update_W(T_float, U_cur, V_cur, R)
                
                err = _compute_error(U_cur, V_cur, W_cur, T_float)
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
            
            # Try to snap to rationals
            if err < 1e-8:
                for scale in [1, 2, 4, 8]:
                    U_s = np.round(U_cur * scale) / scale
                    V_s = np.round(V_cur * scale) / scale
                    W_s = np.round(W_cur * scale) / scale
                    
                    if _check_decomposition_float(U_s, V_s, W_s, T_float):
                        # Verify exactly
                        valid = True
                        for u in range(U_rows):
                            for v in range(V_rows):
                                for w in range(W_rows):
                                    total = Fraction(0)
                                    for r in range(R):
                                        total += _to_fraction(U_s[u, r]) * _to_fraction(V_s[v, r]) * _to_fraction(W_s[w, r])
                                    if total != Fraction(int(T[u, v, w])):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            best_result = (U_s.tolist(), V_s.tolist(), W_s.tolist())
                            break
            if best_result is not None:
                break
    
    return best_result


def _try_algebraic_elimination_rank16(deadline, n_seeds=20):
    """
    Try to find a rank-16 decomposition using algebraic elimination.
    """
    n, m, p = DIMENSIONS
    R = 16
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    T = _build_target()
    T_float = T.astype(np.float64)
    
    U18, V18, W18 = _get_rank18_solution()
    
    best_result = None
    
    for seed in range(n_seeds):
        if time.time() > deadline:
            break
        
        np.random.seed(5000 + seed)
        
        # Start from rank-18, drop 2 terms, and try to recover
        drop1, drop2 = np.random.choice(18, 2, replace=False)
        mask = [i for i in range(18) if i not in [drop1, drop2]]
        
        U_init = U18[:, mask].copy() + np.random.randn(U_rows, R) * 0.1
        V_init = V18[:, mask].copy() + np.random.randn(V_rows, R) * 0.1
        W_init = W18[:, mask].copy() + np.random.randn(W_rows, R) * 0.1
        
        U_cur, V_cur, W_cur = U_init, V_init, W_init
        prev_err = np.inf
        
        for it in range(80):
            if time.time() > deadline:
                break
            
            U_cur = _als_update_U(T_float, V_cur, W_cur, R)
            V_cur = _als_update_V(T_float, U_cur, W_cur, R)
            W_cur = _als_update_W(T_float, U_cur, V_cur, R)
            
            err = _compute_error(U_cur, V_cur, W_cur, T_float)
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        if err < 1e-8:
            for scale in [1, 2, 4]:
                U_s = np.round(U_cur * scale) / scale
                V_s = np.round(V_cur * scale) / scale
                W_s = np.round(W_cur * scale) / scale
                
                if _check_decomposition_float(U_s, V_s, W_s, T_float):
                    valid = True
                    for u in range(U_rows):
                        for v in range(V_rows):
                            for w in range(W_rows):
                                total = Fraction(0)
                                for r in range(R):
                                    total += _to_fraction(U_s[u, r]) * _to_fraction(V_s[v, r]) * _to_fraction(W_s[w, r])
                                if total != Fraction(int(T[u, v, w])):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_result = (U_s.tolist(), V_s.tolist(), W_s.tolist())
                        break
        if best_result is not None:
            break
    
    return best_result


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_target()
    T_float = T.astype(np.float64)
    
    # Baseline: Rank 18
    U18, V18, W18 = _get_rank18_solution()
    best_result = {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
    best_rank = 18
    
    # Try algebraic elimination for rank 17
    search_deadline_17 = min(deadline - 30, start_time + 100)
    result = _try_algebraic_elimination_rank17(search_deadline_17, n_seeds=100)
    if result is not None:
        U_l, V_l, W_l = result
        best_result = {"U": U_l, "V": V_l, "W": W_l}
        best_rank = 17
    
    # Also try ALS-based rank 17 search
    if best_rank == 18 and time.time() < deadline - 25:
        als_deadline = min(deadline - 25, start_time + 110)
        result = _try_als_rank(17, T, als_deadline, n_seeds=50, max_iters=40)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l, "V": V_l, "W": W_l}
            best_rank = 17
    
    # If time permits, try Rank 16
    if time.time() < deadline - 10:
        result = _try_algebraic_elimination_rank16(deadline - 5, n_seeds=15)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l, "V": V_l, "W": W_l}
            best_rank = 16
    
    # Final ALS attempt for rank 16 if not found
    if best_rank > 16 and time.time() < deadline - 3:
        result = _try_als_rank(16, T, deadline - 2, n_seeds=8, max_iters=25)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l, "V": V_l, "W": W_l}
    
    return best_result
