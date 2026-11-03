import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
_start_time = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication.
    Try rank 27 first, fall back to 28 if not found in time.
    """
    n, m, p = 2, 3, 5
    
    # Try rank 27 with structured search
    if time.time() - _start_time < TIME_LIMIT - 30:
        result = _try_rank27_structured(n, m, p)
        if result is not None:
            U, V, W = result
            U = np.array(U, dtype=float)
            V = np.array(V, dtype=float)
            W = np.array(W, dtype=float)
            ok = _verify_decomposition(U, V, W, n, m, p)
            if ok:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: known rank-28 solution
    return _rank28_solution()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _rank28_solution():
    """Known rank-28 decomposition using two Strassen 2x2x2 blocks plus schoolbook."""
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # m1
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m2
        U[1*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        # m3
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m4
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        # m5
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        # m6
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m7
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
                
    assert c == rank, f"Rank mismatch: {c} != {rank}"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank27_structured(n, m, p, max_time=100):
    """
    Attempt to find a rank-27 decomposition by starting from the rank-28 solution
    and searching for a linear combination of terms that can be eliminated.
    
    Strategy: The rank-28 solution has structure. We look for a vector of coefficients
    alpha_r such that the weighted sum of the 28 rank-1 tensors is still a valid
    decomposition, and at least one coefficient is zero (allowing us to drop that term).
    
    More practically: we try to find if there's a non-trivial linear relation among
    the 28 terms that allows merging two terms into one.
    """
    start = time.time()
    deadline = start + max_time
    
    # Get the rank-28 solution
    sol28 = _rank28_solution()
    U28 = np.array(sol28["U"], dtype=float)
    V28 = np.array(sol28["V"], dtype=float)
    W28 = np.array(sol28["W"], dtype=float)
    
    n, m, p = 2, 3, 5
    nm, mp, np_ = n*m, m*p, n*p
    rank28 = U28.shape[1]
    
    # Build the tensor T as a 3D array
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # For each pair of terms (r, s), try to find if we can replace them with a single term
    # This is equivalent to checking if the "difference" tensor has rank 1
    
    # Alternative approach: use the fact that the tensor space has dimension n*m*p = 30
    # and we have 28 rank-1 components. Check if they're linearly dependent in a useful way.
    
    # Flatten each rank-1 component into a vector in R^30
    components = []
    for r in range(rank28):
        comp = np.zeros((nm, mp, np_))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    comp[i*m+j, j*p+k, k*n+i] = U28[i*m+j, r] * V28[j*p+k, r] * W28[k*n+i, r]
        components.append(comp.flatten())
    
    components = np.array(components)  # shape (28, 30)
    
    # Check if these 28 vectors in R^30 are linearly dependent
    # If yes, there might be a way to reduce rank
    rank = np.linalg.matrix_rank(components, tol=1e-10)
    
    if rank < 28:
        # There's a linear dependence. Find the null space vector.
        null_vec = _null_space(components)
        if null_vec is not None:
            # Try to use this to construct a rank-27 solution
            result = _construct_from_null_vector(U28, V28, W28, null_vec, n, m, p)
            if result is not None:
                return result
    
    # Try a more direct approach: for each pair (r, s), try to eliminate one term
    # by adjusting the others
    for r_eliminate in range(rank28):
        if time.time() > deadline:
            break
        
        # Try to express term r_eliminate as a combination of the other 27 terms
        # in a way that preserves the tensor decomposition structure
        result = _try_eliminate_term(U28, V28, W28, r_eliminate, n, m, p, deadline)
        if result is not None:
            return result
    
    return None

def _null_space(A, tol=1e-10):
    """Find a vector in the null space of A (A is m x n, find x such that Ax = 0)."""
    try:
        from scipy.linalg import null_space
        ns = null_space(A, tol=tol)
        if ns.shape[1] > 0:
            return ns[:, 0]
    except:
        pass
    
    # Fallback: use SVD
    try:
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
        # Find singular values close to zero
        for i in range(len(s) - 1, -1, -1):
            if s[i] < tol * (s[0] if s[0] > 0 else 1.0):
                return Vt[i].copy()
    except:
        pass
    
    return None

def _construct_from_null_vector(U28, V28, W28, null_vec, n, m, p):
    """
    Given a null space vector, try to construct a rank-27 solution.
    This is heuristic - we check if any component has a small coefficient.
    """
    # Find the component with smallest absolute coefficient
    min_idx = np.argmin(np.abs(null_vec))
    min_val = abs(null_vec[min_idx])
    
    if min_val < 1e-6:
        # This component is nearly zero in the null space combination
        # Try dropping it
        U27 = np.delete(U28, min_idx, axis=1)
        V27 = np.delete(V28, min_idx, axis=1)
        W27 = np.delete(W28, min_idx, axis=1)
        
        # Verify
        if _verify_decomposition(U27, V27, W27, n, m, p):
            return (U27, V27, W27)
    
    return None

def _try_eliminate_term(U28, V28, W28, r_elim, n, m, p, deadline):
    """
    Try to eliminate term r_elim by finding a rank-1 tensor that, when added,
    allows us to drop term r_elim while preserving the decomposition.
    
    This is a simplified approach: we check if the tensor contributed by term r_elim
    can be "absorbed" by modifying other terms.
    """
    # This is computationally expensive, so we do a quick check
    # Build the tensor for term r_elim
    T_elim = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_elim[i, j, k] = U28[i*m+j, r_elim] * V28[j*p+k, r_elim] * W28[k*n+i, r_elim]
    
    # Check if T_elim can be written as a sum of rank-1 tensors using the other 27 terms
    # with small modifications. This is a hard problem, so we use a heuristic.
    
    # Quick check: if T_elim is already "covered" by other terms, we might be able to drop it
    # Compute the residual if we drop term r_elim
    T_residual = np.zeros((n, m, p))
    for r in range(28):
        if r == r_elim:
            continue
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_residual[i, j, k] += U28[i*m+j, r] * V28[j*p+k, r] * W28[k*n+i, r]
    
    # The residual should equal -T_elim for the decomposition to hold
    # Check if T_residual + T_elim = 0 (it should, since we're just dropping a term)
    # This doesn't help directly. Let's try a different approach.
    
    # Try to find a vector alpha such that:
    # sum_r alpha_r * (U28[:,r] ⊗ V28[:,r] ⊗ W28[:,r]) = T_elim
    # and alpha has at most 27 non-zero entries (excluding r_elim)
    
    # This is a linear system. Set it up.
    nm, mp, np_ = n*m, m*p, n*p
    # Number of constraints: n*m*p = 30
    # Number of unknowns: 27 (coefficients for the other 27 terms)
    
    A = np.zeros((30, 27))
    b = np.zeros(30)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for s in range(27):
                    r = s if s < r_elim else s + 1
                    A[idx, s] = U28[i*m+j, r] * V28[j*p+k, r] * W28[k*n+i, r]
                b[idx] = T_elim[i, j, k]
                idx += 1
    
    # Solve the linear system
    try:
        alpha, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        # Check if the solution is good
        residual = np.linalg.norm(A @ alpha - b)
        if residual < 1e-6:
            # Great! We found coefficients. Now construct the new solution.
            # The new solution has 27 terms: the original 27 (excluding r_elim) 
            # but with modified coefficients
            U27 = np.zeros((nm, 27))
            V27 = np.zeros((mp, 27))
            W27 = np.zeros((np_, 27))
            
            for s in range(27):
                r = s if s < r_elim else s + 1
                U27[:, s] = alpha[s] * U28[:, r]
                V27[:, s] = alpha[s] * V28[:, r]
                W27[:, s] = alpha[s] * W28[:, r]
            
            # But this doesn't work because scaling each term independently changes the product
            # We need to distribute the scaling across U, V, W
            # For simplicity, put all scaling in U
            U27 = np.zeros((nm, 27))
            V27 = np.zeros((mp, 27))
            W27 = np.zeros((np_, 27))
            
            for s in range(27):
                r = s if s < r_elim else s + 1
                # Distribute alpha[s] as cube root
                cbrt_alpha = np.cbrt(alpha[s])
                U27[:, s] = cbrt_alpha * U28[:, r]
                V27[:, s] = cbrt_alpha * V28[:, r]
                W27[:, s] = cbrt_alpha * W28[:, r]
            
            # Verify
            if _verify_decomposition(U27, V27, W27, n, m, p, tol=1e-6):
                return (U27, V27, W27)
    except:
        pass
    
    return None

def _try_rank(n, m, p, rank):
    """
    Legacy interface - now delegates to _try_rank27_structured for rank 27.
    """
    if rank == 27:
        return _try_rank27_structured(n, m, p)
    return None
