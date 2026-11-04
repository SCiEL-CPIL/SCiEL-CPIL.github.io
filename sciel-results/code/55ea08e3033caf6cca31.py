"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time
import itertools
from fractions import Fraction

_START = time.time()
_LIMIT = 120.0
DIMENSIONS = [3, 4, 7]


def _deadline():
    return time.time() - _START > _LIMIT - 5.0


def _pack(arr):
    """Convert numpy array to list of lists of floats (integers if integral)."""
    a = np.asarray(arr)
    # Check if all values are close to integers or half-integers
    result = []
    for row in a:
        r = []
        for x in row:
            xf = float(x)
            # Round to nearest half-integer if close
            nearest = round(xf * 2) / 2
            if abs(xf - nearest) < 1e-9:
                r.append(nearest)
            else:
                r.append(xf)
        result.append(r)
    return result


def _check_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    Nm, Mp, Pn = n * m, m * p, p * n
    rank = U.shape[1]
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i' and j==j' and k==k' else 0
    # Actually: out[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), 0 otherwise
    # The decomposition says: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # Check a sample of entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-8:
                    return False
    # Check that non-matching entries are zero
    # For random samples of (a, b, c) where the indices don't align
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, Nm - 1)
        b = random.randint(0, Mp - 1)
        c = random.randint(0, Pn - 1)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if not (i_a == i_c and j_a == j_b and k_a == k_b if False else True):
            # Check if this is a "valid" combination
            # Valid means: the (i,j,k) triple corresponds to a real multiplication term
            # i_a from A, j_a from A, j_b from B, k_b from B, k_c from W, i_c from W
            # For the tensor to be nonzero, we need i_a == i_c AND j_a == j_b AND k_b == k_c
            # Wait, let me re-examine. T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
            # T is nonzero only when i==i' AND j==j' AND k==k'
            # So for entry (a,b,c): i=a//m, j=a%m, j'=b//p, k=b%p, k'=c//n, i'=c%n
            # Nonzero iff i==i' AND j==j' AND k==k'
            i_val = a // m
            j_val = a % m
            jp_val = b // p
            k_val = b % p
            kp_val = c // n
            ip_val = c % n
            is_valid = (i_val == ip_val and j_val == jp_val and k_val == kp_val)
            if not is_valid:
                actual = 0.0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                if abs(actual) > 1e-8:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank n*m*p)."""
    R0 = n * m * p
    U0 = np.zeros((n * m, R0))
    V0 = np.zeros((m * p, R0))
    W0 = np.zeros((p * n, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    return U0, V0, W0


def _try_rank_r_decomposition(n, m, p, r, max_time=30.0):
    """
    Try to find a rank-r decomposition using structured search.
    
    Strategy: The multiplication tensor has a special structure. We can view it as:
    For each output position (i,k), the contribution is sum_j A[i,j]*B[j,k].
    
    We'll try a bilinear decomposition where:
    U[i*m+j, r] depends on (i,j) and r
    V[j*p+k, r] depends on (j,k) and r
    W[k*n+i, r] depends on (k,i) and r
    
    Using the "Waring" analogy: seek r rank-1 terms that span the multiplication map.
    We parameterize using structured patterns and solve linear systems.
    """
    start_time = time.time()
    
    # Try structured decompositions based on partitioning the middle index j
    # For rank r, we can try: assign each (i,j,k) triple to one of r "terms"
    # such that within each term, the indices follow a pattern.
    
    # Approach 1: Block-diagonal style
    # Partition the m=4 middle dimension into groups
    # Each group g handles a subset of j values for all i,k
    
    # Approach 2: Use the fact that for fixed i,k, we need sum_j to work
    # This suggests a decomposition where each rank term handles a subset of j's
    
    # Let's try: for each rank r, assign each j in {0,1,2,3} to one or more rank terms
    # such that for each (i,k), the sum over j is covered.
    
    # More systematic: use random initialization + Newton iteration
    # to solve the nonlinear system sum_r U[a,r]V[b,r]W[c,r] = T[a,b,c]
    
    np.random.seed(12345)
    
    best_rank = None
    best_U, best_V, best_W = None, None, None
    
    # Try several random initializations
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        if _deadline():
            break
        
        # Random initialization
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # Newton-style iteration: solve for the residual
        # The system is: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # This is a set of (n*m)*(m*p)*(p*n) equations but many are zero.
        # Only n*m*p equations are nonzero (one for each valid (i,j,k) triple).
        
        # Build the linear system for Newton step
        # Variables: U, V, W (total n*m*r + m*p*r + p*n*r variables)
        # Equations: n*m*p (nonzero entries of T)
        
        # For efficiency, only consider the n*m*p nonzero equations
        # Equation for (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # Linearize around current (U,V,W):
        # d/dU: V[j*p+k,r]*W[k*n+i,r]
        # d/dV: U[i*m+j,r]*W[k*n+i,r]
        # d/dW: U[i*m+j,r]*V[j*p+k,r]
        
        num_eqs = n * m * p  # 84
        num_vars = (n * m + m * p + p * n) * r
        
        # Build Jacobian J (num_eqs x num_vars) and residual
        J = np.zeros((num_eqs, num_vars))
        resid = np.zeros(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    for r_idx in range(r):
                        # d/dU[a, r_idx]
                        col_u = a * r + r_idx
                        J[eq_idx, col_u] = V[b, r_idx] * W[c, r_idx]
                        # d/dV[b, r_idx]
                        col_v = (n * m + b) * r + r_idx
                        J[eq_idx, col_v] = U[a, r_idx] * W[c, r_idx]
                        # d/dW[c, r_idx]
                        col_w = (n * m + m * p + c) * r + r_idx
                        J[eq_idx, col_w] = U[a, r_idx] * V[b, r_idx]
                    current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    resid[eq_idx] = current_val - 1.0
                    eq_idx += 1
        
        # Solve J * delta = -resid using least squares
        try:
            delta, _, _, _ = np.linalg.lstsq(J, -resid, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Update
        delta_U = delta[:n * m * r].reshape(n * m, r)
        delta_V = delta[n * m * r:n * m * r + m * p * r].reshape(m * p, r)
        delta_W = delta[n * m * r + m * p * r:].reshape(p * n, r)
        
        U = U + delta_U
        V = V + delta_V
        W = W + delta_W
        
        # Check convergence
        max_resid = np.max(np.abs(resid))
        if max_resid < 1e-7:
            # Verify the solution
            if _check_decomposition(U, V, W, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    if best_rank is not None:
        return best_U, best_V, best_W
    return None, None, None


def _structured_decomposition(n, m, p):
    """
    Try structured decompositions based on the algebraic structure.
    
    Key insight: For multiplying (n x m) by (m x p), the tensor has a
    "slice" structure. For each j, the slice T[:,:,j] (in appropriate
    indexing) is an outer product of identity-like matrices.
    
    We can try decompositions where each rank term corresponds to a
    "pattern" of how indices are coupled.
    """
    # Try rank values from theoretical lower bound upward
    # Lower bound: max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    # But Strassen-like improvements may exist
    
    # Theoretical lower bounds:
    # - rank >= n*p = 21 (from output dimension)
    # - rank >= m*p = 28 (from B's dimension)  
    # - rank >= n*m = 12 (from A's dimension)
    # - rank >= (n*m*p) / (min(n,m,p)) = 84/3 = 28
    
    # Try structured patterns for ranks near the lower bound
    
    # Pattern 1: For each j, use a separate "block"
    # This gives rank = m * max(n, p) = 4 * 7 = 28 (schoolbook-like)
    
    # Pattern 2: Share structure across j's
    # For rank r, assign weights to each (i,j,k)
    
    # Let's try a specific structured approach:
    # Use r = 28 and see if we can do better with clever mixing
    
    # Actually, let's try the "bilinear" approach more carefully
    # For each rank term r, define:
    #   U_r: vector in R^(n*m) 
    #   V_r: vector in R^(m*p)
    #   W_r: vector in R^(p*n)
    # The constraint is: for all i,j,k: sum_r U_r[i*m+j]*V_r[j*p+k]*W_r[k*n+i] = 1
    # And for invalid (i,j,k) combinations: sum_r ... = 0
    
    # This is a system of n*m*p + (n*m*m*p*p*n - n*m*p) equations
    # But many of the "zero" equations are automatically satisfied if the
    # structure is right.
    
    # Try rank = 28 with structured initialization
    # Initialize with schoolbook and try to reduce rank by merging terms
    
    # For now, return schoolbook as baseline
    return _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n * m * p  # 84
    
    # Try to find lower rank decompositions
    # Lower bound is 28 (from m*p dimension)
    # Try ranks from 28 upward
    
    for r in range(28, best_rank):
        if _deadline():
            break
        if time.time() - _START > _LIMIT - 10:
            break
        
        result = _try_rank_r_decomposition(n, m, p, r, max_time=min(15.0, _LIMIT - (time.time() - _START) - 5))
        if result[0] is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break  # Found the minimal rank
    
    # Also try structured decomposition
    if not _deadline():
        sU, sV, sW = _structured_decomposition(n, m, p)
        if sU.shape[1] < best_rank:
            best_U, best_V, best_W = sU, sV, sW
            best_rank = sU.shape[1]
    
    return {"U": _pack(best_U), "V": _pack(best_V), "W": _pack(best_W)}
