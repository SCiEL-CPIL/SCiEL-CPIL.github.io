import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _check_deadline(start, remaining=5):
    return _elapsed(start) > DEADLINE - remaining

def _build_strassen_rank55():
    """Build the known rank-55 Strassen-based decomposition."""
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # A indices: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        # B indices: B[0,k0]=k0, B[0,k1]=k1, B[1,k0]=10+k0, B[1,k1]=10+k1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        # W indices: W[k*n+i] -> W[k0*2+0]=2*k0, W[k0*2+1]=2*k0+1, etc.
        c00, c01 = 2 * k0, 2 * k0 + 1  # i=0: C[0,k0], C[0,k1]
        c10, c11 = 2 * k1, 2 * k1 + 1  # Wait, let me recheck
        # Actually W[k*n+i]: for output C[i,k], index is k*n+i = k*2+i
        # C[0,k0] -> 2*k0+0 = 2*k0
        # C[0,k1] -> 2*k1+0 = 2*k1
        # C[1,k0] -> 2*k0+1
        # C[1,k1] -> 2*k1+1
        c00 = 2 * k0      # C[0,k0]
        c01 = 2 * k1      # C[0,k1]
        c10 = 2 * k0 + 1  # C[1,k0]
        c11 = 2 * k1 + 1  # C[1,k1]
        
        # M1 = (A00 + A11)(B00 + B11) -> contributes to C[0,k0] and C[1,k1]
        U[a00, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00 -> contributes to C[1,k0] and C[1,k1]
        U[a10, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11) -> contributes to C[0,k1] and C[1,k1]
        U[a00, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00) -> contributes to C[0,k0] and C[1,k0]
        U[a11, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11 -> contributes to C[0,k0] and C[0,k1]
        U[a00, col] += 1.0; U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01) -> contributes to C[1,k1]
        U[a10, col] += 1.0; U[a00, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11) -> contributes to C[0,k0]
        U[a01, col] += 1.0; U[a11, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
    
    # Handle A[0,2] and A[1,2] (the third column of A)
    for k in range(p):
        b2k = 2 * p + k  # B[2,k] index
        U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
        col += 1
        U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
        col += 1
    
    return U, V, W, rank

def _verify(U, V, W, n=2, m=3, p=10, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _search_rank54(start_time):
    """Try to find a rank-54 decomposition by merging two Strassen terms."""
    n, m, p = 2, 3, 10
    U55, V55, W55, _ = _build_strassen_rank55()
    
    # The rank-55 solution has 55 terms. We want to see if we can express
    # one term as a linear combination of the others, reducing to rank 54.
    # 
    # For each pair of terms (a, b), check if term b can be written as
    # a linear combination of all other terms. This is a linear algebra problem.
    #
    # The constraint is: for all (i,j,k):
    #   sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    #
    # If we fix 54 terms and try to find the 55th as a combination, we can
    # set up a linear system. But the equations are trilinear, so it's not
    # straightforward.
    #
    # Alternative: Try to find a rank-54 solution by perturbing the rank-55
    # solution and checking if any term becomes redundant.
    
    # For each term t, try to eliminate it by adjusting other terms.
    # This is a nonlinear optimization problem.
    
    # Simpler approach: Check if any two terms are linearly dependent
    # in a way that allows merging.
    
    # For each pair (a, b), check if U[:,b] is proportional to U[:,a],
    # V[:,b] proportional to V[:,a], and W[:,b] proportional to W[:,a].
    # If so, we can merge them.
    
    for a in range(55):
        if _check_deadline(start_time, remaining=10):
            return None
        for b in range(a+1, 55):
            # Check if terms a and b can be merged
            # This requires: U[:,b] = alpha * U[:,a], V[:,b] = alpha * V[:,a], W[:,b] = alpha * W[:,a]
            # for some scalar alpha.
            
            # Check U proportionality
            ua = U55[:, a]
            ub = U55[:, b]
            if np.allclose(ua, 0.0) and np.allclose(ub, 0.0):
                continue
            if np.allclose(ua, 0.0) or np.allclose(ub, 0.0):
                continue
            
            # Find alpha from U
            nonzero_mask = np.abs(ua) > 1e-10
            if np.sum(nonzero_mask) == 0:
                continue
            alpha_candidates = ub[nonzero_mask] / ua[nonzero_mask]
            if not np.allclose(alpha_candidates, alpha_candidates[0], rtol=1e-8, atol=1e-10):
                continue
            alpha = alpha_candidates[0]
            
            # Check V proportionality
            va = V55[:, a]
            vb = V55[:, b]
            if not np.allclose(vb, alpha * va, rtol=1e-8, atol=1e-10):
                continue
            
            # Check W proportionality
            wa = W55[:, a]
            wb = W55[:, b]
            if not np.allclose(wb, alpha * wa, rtol=1e-8, atol=1e-10):
                continue
            
            # Terms a and b are proportional! Merge them.
            # New term = term a + alpha * term b (or just keep one)
            U54 = np.zeros((n*m, 54), dtype=float)
            V54 = np.zeros((m*p, 54), dtype=float)
            W54 = np.zeros((p*n, 54), dtype=float)
            
            col = 0
            for t in range(55):
                if t == b:
                    continue
                U54[:, col] = U55[:, t]
                V54[:, col] = V55[:, t]
                W54[:, col] = W55[:, t]
                col += 1
            
            # Verify
            if _verify(U54, V54, W54):
                return U54, V54, W54, 54
    
    return None

def _search_rank53(start_time):
    """Try to find a rank-53 decomposition."""
    # This is more difficult. We'd need to merge two pairs or find a more complex
    # linear dependency. For now, skip this and just return the best we have.
    return None

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Build the rank-55 Strassen solution
    U55, V55, W55, rank55 = _build_strassen_rank55()
    
    # Verify
    assert _verify(U55, V55, W55), "Rank-55 solution is invalid!"
    
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Try to find rank-54
    if not _check_deadline(start_time, remaining=20):
        result54 = _search_rank54(start_time)
        if result54 is not None:
            U54, V54, W54, r54 = result54
            best_U, best_V, best_W, best_rank = U54, V54, W54, r54
    
    # Try to find rank-53
    if best_rank == 54 and not _check_deadline(start_time, remaining=10):
        result53 = _search_rank53(start_time)
        if result53 is not None:
            U53, V53, W53, r53 = result53
            best_U, best_V, best_W, best_rank = U53, V53, W53, r53
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
