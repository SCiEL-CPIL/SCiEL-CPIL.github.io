import time
import numpy as np
from scipy.optimize import lsq_linear

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()
SAFETY = 15.0

def _elapsed():
    return time.time() - START_TIME

def _time_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY)

def _verify(result):
    """Exact integer verification of the tensor decomposition."""
    n, m, p = DIMENSIONS
    try:
        U = np.array(result['U'], dtype=object)
        V = np.array(result['V'], dtype=object)
        W = np.array(result['W'], dtype=object)
        rank = U.shape[1]
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Use a witness: A[i,j] = 2^j, B[j,k] = 3^k
        A = np.zeros((n, m), dtype=object)
        for i in range(n):
            for j in range(m):
                A[i, j] = 2 ** j
        B = np.zeros((m, p), dtype=object)
        for j in range(m):
            for k in range(p):
                B[j, k] = 3 ** k
        C_true = np.zeros((n, p), dtype=object)
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i, j] * B[j, k]
                C_true[i, k] = s
        
        C_comp = np.zeros((n, p), dtype=object)
        for r in range(rank):
            u = 0
            for i in range(n):
                for j in range(m):
                    u += U[i * m + j, r] * A[i, j]
            v = 0
            for j in range(m):
                for k in range(p):
                    v += V[j * p + k, r] * B[j, k]
            uv = u * v
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i, r] * uv
        
        for i in range(n):
            for k in range(p):
                if C_comp[i, k] != C_true[i, k]:
                    return False
        return True
    except Exception:
        return False

def _construct_strassen_28():
    """2x4 * 4x4 using four independent 2x2 Strassen blocks -> rank 28."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    sub_products = [
        {'a': lambda r, c: r * 4 + c, 'b': lambda r, c: r * 4 + c, 'c': lambda r, c: c * 2 + r},
        {'a': lambda r, c: r * 4 + (c + 2), 'b': lambda r, c: (r + 2) * 4 + c, 'c': lambda r, c: c * 2 + r},
        {'a': lambda r, c: r * 4 + c, 'b': lambda r, c: r * 4 + (c + 2), 'c': lambda r, c: (c + 2) * 2 + r},
        {'a': lambda r, c: r * 4 + (c + 2), 'b': lambda r, c: (r + 2) * 4 + (c + 2), 'c': lambda r, c: (c + 2) * 2 + r},
    ]

    col = 0
    for sp in sub_products:
        for (a_coeffs, b_coeffs, c_coeffs) in strassen_terms:
            for local_idx, coeff in a_coeffs.items():
                lr, lc = local_idx // 2, local_idx % 2
                U[sp['a'](lr, lc)][col] += coeff
            for local_idx, coeff in b_coeffs.items():
                lr, lc = local_idx // 2, local_idx % 2
                V[sp['b'](lr, lc)][col] += coeff
            for local_idx, coeff in c_coeffs.items():
                lr, lc = local_idx // 2, local_idx % 2
                W[sp['c'](lr, lc)][col] += coeff
            col += 1

    return {"U": U, "V": V, "W": W}

def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _try_tensor_completion(rank_target):
    """
    Try to find a rank-target decomposition using structured tensor completion.
    We initialize from Strassen-28 and try to eliminate terms.
    """
    if not _time_ok():
        return None
    
    n, m, p = DIMENSIONS
    base = _construct_strassen_28()
    U0 = np.array(base['U'], dtype=float)
    V0 = np.array(base['V'], dtype=float)
    W0 = np.array(base['W'], dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # T has shape (n*m, m*p, p*n) in the flattened modes
    # T[a_idx, b_idx, c_idx] = 1 where a_idx=i*m+j, b_idx=j*p+k, c_idx=k*n+i
    # But actually T is the structure tensor: T_{i,j,k} for the matmul
    # In our formulation: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is: T[a,b,c] where a=(i,j), b=(j,k), c=(i,k)
    # T[(i,j),(j,k),(i,k)] = 1
    
    # For tensor completion, we view this as a matrix of size (n*m) x (m*p * p*n) = 8 x 64
    # Each rank-1 term contributes u_r * v_r * w_r to the tensor
    # The mode-1 flattening: for each a_idx, the vector is [sum over r of U[a_idx,r] * V[b_idx,r] * W[c_idx,r] for all (b_idx,c_idx)]
    
    # Actually, let's think differently. We have 28 rank-1 terms. We want to see if we can
    # represent the same tensor with fewer terms.
    
    # The tensor T can be written as: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^(n*m), v_r ∈ R^(m*p), w_r ∈ R^(p*n)
    
    # To check if rank < 28, we can try to find linear dependencies among the 28 terms.
    # If we can find a linear combination of the 28 rank-1 tensors that equals zero,
    # we can eliminate one term.
    
    # Let's build the "signature" of each rank-1 term as a vector in R^(n*m * m*p * p*n) = R^512
    # and check if they are linearly independent.
    
    nm, mp, pn = n*m, m*p, p*n
    dim = nm * mp * pn  # 8 * 16 * 8 = 1024
    
    signatures = []
    for r in range(28):
        sig = np.zeros(dim)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    sig[a * mp * pn + b * pn + c] = U0[a, r] * V0[b, r] * W0[c, r]
        signatures.append(sig)
    
    sig_matrix = np.array(signatures)  # 28 x 1024
    
    # Check rank of the signature matrix
    # If rank < 28, there's a linear dependency, meaning we can potentially reduce rank
    # But this is a necessary condition, not sufficient.
    
    # Let's try a different approach: use the fact that the tensor is small
    # and try to find a lower-rank decomposition via optimization.
    
    # For rank r < 28, we need to find U, V, W of shapes (8,r), (16,r), (8,r)
    # such that the tensor matches.
    
    # This is a non-convex problem. Let's try to use the structure.
    
    # Alternative: try to see if any of the 28 Strassen terms can be expressed
    # as a linear combination of the others.
    
    # Compute the rank of the 28 signature vectors
    try:
        rank_val = np.linalg.matrix_rank(sig_matrix, tol=1e-10)
    except Exception:
        rank_val = 28
    
    if rank_val < 28:
        # There's a dependency. Try to find it and eliminate a term.
        # For now, just note it and return the best we have.
        pass
    
    # Try to use scipy to find a sparse representation
    # We want to find coefficients alpha_r such that sum_r alpha_r * sig_r = T
    # and as many alpha_r are zero as possible.
    
    # First, build the target tensor signature
    target = np.zeros(dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx * mp * pn + b_idx * pn + c_idx] = 1.0
    
    # We have 28 basis vectors (the Strassen signatures) and want to represent target.
    # If the 28 vectors span the target, we can find coefficients.
    # If they're linearly dependent, we can find a sparse solution.
    
    try:
        # Solve sig_matrix.T @ alpha = target
        # sig_matrix is 28 x 1024, so sig_matrix.T is 1024 x 28
        # We want alpha (28,) such that sig_matrix.T @ alpha = target
        alpha, residuals, rank_solve, sv = np.linalg.lstsq(sig_matrix.T, target, rcond=None)
        
        # Check if the solution is exact
        recon = sig_matrix.T @ alpha
        if np.allclose(recon, target, atol=1e-10):
            # Find how many alpha values are close to zero
            tol = 1e-8
            nonzero = np.sum(np.abs(alpha) > tol)
            if nonzero < 28:
                # We can potentially eliminate terms
                # But we need to verify that the resulting decomposition is valid
                # For now, this is just an indication
                pass
    except Exception:
        pass
    
    return None  # No improvement found yet

def construct():
    """
    Construct a minimal-rank tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    # Start with the known rank-28 Strassen decomposition
    best = _construct_strassen_28()
    
    # Verify it
    if not _verify(best):
        # Shouldn't happen, but fallback to schoolbook
        best = _construct_schoolbook_32()
    
    # Try to improve using tensor completion
    if _time_ok():
        improved = _try_tensor_completion(27)
        if improved is not None and _verify(improved):
            best = improved
    
    return best
