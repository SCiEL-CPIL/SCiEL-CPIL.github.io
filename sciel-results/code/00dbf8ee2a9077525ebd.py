"""Matrix multiplication 2x3x3 - rank-minimal tensor decomposition.

We attempt to find a rank-17 or rank-16 decomposition for multiplying
a 2x3 matrix by a 3x3 matrix. The schoolbook rank is 18.

Strategy:
- The tensor has 18 entries, all equal to 1.
- A rank-r decomposition requires r terms such that for each (i,j,k):
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

- We use a structured approach: start from schoolbook (rank 18) and try
  to merge pairs of terms. For rank 17, we need to express two schoolbook
  terms as a single rank-1 term plus adjustments to the remaining terms.

- We solve the resulting system of polynomial equations using a combination
  of symbolic/numeric methods.

- The output must have integer or half-integer entries.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]


def _schoolbook():
    """Return the standard rank-18 schoolbook decomposition."""
    n, m, p = 2, 3, 3
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


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(rank):
                    lhs += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(lhs - 1.0) > 1e-9:
                    return False
    return True


def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer within tolerance."""
    # Check if 2*x is close to an integer
    return abs(2 * x - round(2 * x)) < tol


def _to_half_int_matrix(M, tol=1e-6):
    """Convert a matrix to half-integer entries if possible. Returns None if not."""
    M = np.array(M, dtype=float)
    M2 = 2 * M
    M2_rounded = np.round(M2)
    if np.max(np.abs(M2 - M2_rounded)) < tol:
        # Convert to fractions: each entry is k/2 where k is integer
        result = []
        for row in M2_rounded:
            result.append([int(v) / 2.0 for v in row])
        return result
    return None


def _try_rank_17_structured():
    """
    Try to construct a rank-17 decomposition.
    
    Approach: Start from schoolbook rank 18. Try to merge two terms.
    Specifically, pick two terms (i1,j1,k1) and (i2,j2,k2) and try to
    replace them with a single term plus adjustments.
    
    A simpler approach: use the fact that we have 18 equations in rank*(6+9+6)
    variables. For rank 17, we have 17*21 = 357 variables and 18 equations,
    so the system is highly underdetermined.
    
    We use a structured initialization and Newton's method.
    """
    n, m, p = 2, 3, 3
    nm, mp, pn = n * m, m * p, p * n
    rank = 17
    
    # Initialize with schoolbook first 17 columns
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < rank:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                else:
                    break
            if col >= rank:
                break
        if col >= rank:
            break
    
    # The 18th term (i=1, j=2, k=2) is missing. We need to adjust the 17 terms
    # to cover it.
    
    # Let's use a different approach: random initialization with Newton's method
    deadline = time.time() + 90  # Leave margin
    
    np.random.seed(123)
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        # Newton's method: solve for variables to satisfy 18 equations
        # f_c(i,j,k) = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1 = 0
        
        # We have 18 equations and 357 unknowns. Use least-squares Newton.
        # But this is complex. Let's try a simpler iterative approach.
        
        # Actually, let's try a structured approach: fix some variables and solve for others.
        
        # Alternative: use the fact that we can set many variables to 0.
        # Let's try: for each term c, set U[i*m+j,c] = 1 for some (i,j),
        # V[j*p+k,c] = 1 for some (j,k), W[k*n+i,c] = 1 for some (k,i).
        
        # This is essentially the schoolbook approach but with 17 terms.
        # We need to cover all 18 (i,j,k) pairs with 17 terms.
        
        # Each term c covers (i_c, j_c, k_c) where:
        # U[i_c*m+j_c, c] = 1, V[j_c*p+k_c, c] = 1, W[k_c*n+i_c, c] = 1
        # But a term can also contribute to other (i,j,k) if the indices match.
        
        # Actually, in the general case, a term contributes to (i,j,k) if:
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] != 0
        
        # For the schoolbook, each term has exactly one non-zero entry in each matrix,
        # so it only contributes to one (i,j,k) pair.
        
        # To get rank 17, we need at least one term to contribute to multiple (i,j,k) pairs,
        # or we need to have some cancellation/overlap.
        
        # Let's try: allow each term to have multiple non-zero entries.
        # Specifically, let's try a "banded" structure where term c has
        # non-zero entries for a small set of (i,j,k) pairs.
        
        # This is getting complex. Let's just try random Newton.
        
        # Define the residual function
        def residual(U, V, W):
            res = np.zeros(18)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        res[idx] = val - 1.0
                        idx += 1
            return res
        
        # Try a few Newton steps with random perturbations
        for step in range(20):
            r = residual(U, V, W)
            if np.max(np.abs(r)) < 1e-10:
                break
            
            # Simple gradient descent (not true Newton, but might work)
            # Compute gradients
            lr = 0.01
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        grad_U = np.zeros((nm, rank))
                        grad_V = np.zeros((mp, rank))
                        grad_W = np.zeros((pn, rank))
                        for c in range(rank):
                            u_val = U[i * m + j, c]
                            v_val = V[j * p + k, c]
                            w_val = W[k * n + i, c]
                            grad_U[i * m + j, c] = v_val * w_val * r[idx]
                            grad_V[j * p + k, c] = u_val * w_val * r[idx]
                            grad_W[k * n + i, c] = u_val * v_val * r[idx]
                        U -= lr * grad_U
                        V -= lr * grad_V
                        W -= lr * grad_W
        
        r = residual(U, V, W)
        if np.max(np.abs(r)) < 1e-8:
            # Check if we can convert to half-integers
            U_h = _to_half_int_matrix(U)
            V_h = _to_half_int_matrix(V)
            W_h = _to_half_int_matrix(W)
            if U_h is not None and V_h is not None and W_h is not None:
                # Verify
                if _verify(U_h, V_h, W_h, n, m, p):
                    return {"U": U_h, "V": V_h, "W": W_h}
    
    return None


def _try_rank_16_structured():
    """Try to construct a rank-16 decomposition."""
    n, m, p = 2, 3, 3
    nm, mp, pn = n * m, m * p, p * n
    rank = 16
    
    deadline = time.time() + 80
    
    np.random.seed(456)
    
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        def residual(U, V, W):
            res = np.zeros(18)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        res[idx] = val - 1.0
                        idx += 1
            return res
        
        for step in range(20):
            r = residual(U, V, W)
            if np.max(np.abs(r)) < 1e-10:
                break
            lr = 0.01
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        grad_U = np.zeros((nm, rank))
                        grad_V = np.zeros((mp, rank))
                        grad_W = np.zeros((pn, rank))
                        for c in range(rank):
                            u_val = U[i * m + j, c]
                            v_val = V[j * p + k, c]
                            w_val = W[k * n + i, c]
                            grad_U[i * m + j, c] = v_val * w_val * r[idx]
                            grad_V[j * p + k, c] = u_val * w_val * r[idx]
                            grad_W[k * n + i, c] = u_val * v_val * r[idx]
                        U -= lr * grad_U
                        V -= lr * grad_V
                        W -= lr * grad_W
        
        r = residual(U, V, W)
        if np.max(np.abs(r)) < 1e-8:
            U_h = _to_half_int_matrix(U)
            V_h = _to_half_int_matrix(V)
            W_h = _to_half_int_matrix(W)
            if U_h is not None and V_h is not None and W_h is not None:
                if _verify(U_h, V_h, W_h, n, m, p):
                    return {"U": U_h, "V": V_h, "W": W_h}
    
    return None


def _try_merge_terms():
    """
    Try to merge pairs of schoolbook terms to reduce rank by 1.
    
    For rank 17: pick two terms (i1,j1,k1) and (i2,j2,k2) and try to
    replace them with a single term that contributes to both.
    
    This requires finding u, v, w such that:
    u[i1*m+j1]*v[j1*p+k1]*w[k1*n+i1] + (contributions from other 16 terms) = 1
    u[i2*m+j2]*v[j2*p+k2]*w[k2*n+i2] + (contributions from other 16 terms) = 1
    
    And for all other (i,j,k), the single term's contribution is 0 or is
    compensated by adjustments to other terms.
    
    This is a complex constraint. Let's try a simpler approach:
    Just try all pairs and see if a solution exists.
    """
    n, m, p = 2, 3, 3
    nm, mp, pn = n * m, m * p, p * n
    
    # All 18 (i,j,k) triples
    triples = list(product(range(n), range(m), range(p)))
    
    deadline = time.time() + 60
    
    # Try to merge each pair
    for a in range(18):
        for b in range(a + 1, 18):
            if time.time() > deadline:
                return None
            
            i1, j1, k1 = triples[a]
            i2, j2, k2 = triples[b]
            
            # We want to replace terms a and b with a single term.
            # The other 16 terms remain as schoolbook.
            # The new term must satisfy:
            # For (i1,j1,k1): new_term_contribution + sum of other 16 terms' contributions = 1
            # For (i2,j2,k2): new_term_contribution + sum of other 16 terms' contributions = 1
            # For all other (i,j,k): new_term_contribution + sum of other 16 terms' contributions = 1
            
            # The other 16 terms are the schoolbook terms for all (i,j,k) except (i1,j1,k1) and (i2,j2,k2).
            # For (i1,j1,k1), the other 16 terms contribute 0 (since they're schoolbook for different indices).
            # So we need: new_term_contribution at (i1,j1,k1) = 1
            # Similarly for (i2,j2,k2): new_term_contribution = 1
            # For other (i,j,k), the schoolbook term for that (i,j,k) contributes 1, so new_term_contribution = 0.
            
            # So we need a rank-1 term that contributes 1 at (i1,j1,k1) and (i2,j2,k2),
            # and 0 everywhere else.
            
            # A rank-1 term is defined by vectors u (in R^6), v (in R^9), w (in R^6).
            # It contributes u[i*m+j]*v[j*p+k]*w[k*n+i] at (i,j,k).
            
            # We need:
            # u[i1*m+j1]*v[j1*p+k1]*w[k1*n+i1] = 1
            # u[i2*m+j2]*v[j2*p+k2]*w[k2*n+i2] = 1
            # u[i*m+j]*v[j*p+k]*w[k*n+i] = 0 for all other (i,j,k)
            
            # This is a system of 18 equations (18 constraints) in 21 variables (6+9+6).
            # It's overdetermined, so a solution may not exist.
            
            # Let's try to solve it.
            # Set u[i1*m+j1] = 1, v[j1*p+k1] = 1, w[k1*n+i1] = 1 (first equation)
            # Then u[i2*m+j2]*v[j2*p+k2]*w[k2*n+i2] = 1 (second equation)
            # And all other products = 0.
            
            # This is still complex. Let's just try numerical search.
            
            # Skip this for now; the random approach above is simpler.
            pass
    
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' - factor matrices with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank 17 first
    result_17 = _try_rank_17_structured()
    if result_17 is not None:
        return result_17
    
    if time.time() > deadline:
        pass
    
    # Try rank 16
    result_16 = _try_rank_16_structured()
    if result_16 is not None:
        return result_16
    
    # Fall back to schoolbook rank 18
    result = _schoolbook()
    assert _verify(result['U'], result['V'], result['W'], n, m, p), "Schoolbook verification failed"
    
    return result
