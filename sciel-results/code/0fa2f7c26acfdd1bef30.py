"""Matrix multiplication 2,3,3 - rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 3]

def _to_rational_approx_fast(matrix, max_den=1000):
    """Convert a float matrix to a rational matrix using limit_denominator."""
    result = []
    for row in matrix:
        r = []
        for val in row:
            # Handle zero explicitly to avoid Fraction(0.0) issues or slow paths
            if abs(val) < 1e-12:
                r.append(Fraction(0))
            else:
                # limit_denominator is much faster than manual search
                frac = Fraction(val).limit_denominator(max_den)
                r.append(frac)
        result.append(r)
    return result

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic for ALL entries."""
    rank = len(U[0])
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute target values for all (u, v, w)
    # u in [0, nm), v in [0, mp), w in [0, pn)
    # Target is 1 if there exist i,j,k such that:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # Else 0.
    
    targets = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                targets[(u, v, w)] = 1

    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                target = targets.get((u, v, w), 0)
                s = Fraction(0)
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[w][c]
                if s != target:
                    return False
    return True

def _try_rank(rank, n, m, p, max_attempts=100, time_limit=None):
    """Try to find a rank-r decomposition using full linear system."""
    rng = np.random.default_rng(42 + rank)
    nm = n * m
    mp = m * p
    pn = p * n
    num_eqs = nm * mp * pn
    num_unk = pn * rank
    
    # Precompute indices for the "1" positions
    ones_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                # Flatten index for the equation
                eq_idx = (u * mp + v) * pn + w
                ones_indices.append(eq_idx)
    
    # We will build A and b in each attempt because U and V change.
    # To speed up, we can precompute the structure of A if U and V were fixed, 
    # but they are not.
    
    for attempt in range(max_attempts):
        if time_limit is not None:
            if time.time() > time_limit:
                break
        
        # Random U and V
        # Scale to avoid overflow/underflow
        U = rng.random((nm, rank)) * 2 - 1
        V = rng.random((mp, rank)) * 2 - 1
        
        # Build A (num_eqs x num_unk) and b (num_eqs,)
        # A[eq_idx, unk_idx] = U[u, c] * V[v, c]
        # unk_idx = w * rank + c
        # eq_idx = (u * mp + v) * pn + w
        
        A = np.zeros((num_eqs, num_unk))
        b = np.zeros(num_eqs)
        
        # Fill b for the ones
        for eq_idx in ones_indices:
            b[eq_idx] = 1.0
            
        # Fill A
        # This loop is the bottleneck. 
        # num_eqs = 324, num_unk = 54 (for rank 9).
        # 324 * 54 = 17496 operations. Very fast.
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    eq_idx = (u * mp + v) * pn + w
                    for c in range(rank):
                        unk_idx = w * rank + c
                        A[eq_idx, unk_idx] = U[u, c] * V[v, c]
        
        # Solve A w = b
        try:
            w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if solution is valid
        # Residual should be close to 0
        if len(residuals) > 0:
            if residuals[0] > 1e-7:
                continue
        else:
            # Under-determined or exactly determined, check norm of A*w - b
            diff = np.linalg.norm(A @ w_flat - b)
            if diff > 1e-7:
                continue
        
        # Reshape W
        W = w_flat.reshape((pn, rank))
        
        # Convert to rationals
        # Try small denominators first
        found_rational = False
        for max_den in [10, 50, 100, 500, 1000, 5000]:
            U_r = _to_rational_approx_fast(U, max_den)
            V_r = _to_rational_approx_fast(V, max_den)
            W_r = _to_rational_approx_fast(W, max_den)
            
            # Verify exact
            if _verify_decomposition_exact(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
        
        # If rational conversion failed with these denominators, 
        # it's possible the float solution is valid but not "nice" rational,
        # or the float solution is slightly off due to numerical error.
        # We can try to polish the float solution or accept it if the evaluator
        # accepts floats? No, task says "integer/half-integer entries".
        # So we MUST find a rational solution.
        
        # If we failed to find a rational solution, we continue to next attempt.
        # Note: It is possible that the specific random U, V led to a valid 
        # float decomposition, but the rational entries have large denominators.
        # Trying more attempts with different random U, V might yield a "nicer" one.

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second budget with safety margin
    
    # Try ranks from lower bound to schoolbook
    # Lower bound is max(nm, mp, pn) = 9
    schoolbook_rank = n * m * p
    
    for rank in range(9, schoolbook_rank + 1):
        if time.time() > deadline - 30:  # Keep 30s margin
            break
        
        result = _try_rank(rank, n, m, p, max_attempts=200, time_limit=deadline - 15)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
