"""
Tensor decomposition for 2x3 x 3x7 matrix multiplication with rank minimization.

Strategy:
1. Construct the schoolbook decomposition (rank 42).
2. Verify if the rank-1 terms are linearly independent.
3. If dependent, attempt to find a minimal subset that exactly represents the tensor.
4. Return the best decomposition found.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]
DEADLINE = 145.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def remaining():
    return DEADLINE - elapsed()


def build_target_tensor():
    """Build the standard matrix multiplication tensor for 2x3 * 3x7."""
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_input = i * m + j
                j_input = j * p + k
                k_output = k * n + i
                T[i_input, j_input, k_output] = 1.0
    return T


def schoolbook_decomposition():
    """Generate the trivial schoolbook decomposition of rank 42."""
    n, m, p = 2, 3, 7
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


def verify_decomposition(U, V, W, T=None, tol=1e-9):
    """Verify that the tensor decomposition is correct."""
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.ndim == 1:
        U_arr = U_arr.reshape(nm, -1)
    if V_arr.ndim == 1:
        V_arr = V_arr.reshape(mp, -1)
    if W_arr.ndim == 1:
        W_arr = W_arr.reshape(pn, -1)
    
    rank = U_arr.shape[1]
    
    if T is None:
        T = build_target_tensor()
    
    # Efficient check: sum of outer products
    # T_computed = sum_t U[:,t] outer V[:,t] outer W[:,t]
    # Using einsum for clarity and speed
    T_computed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    max_err = np.max(np.abs(T_computed - T))
    return max_err < tol, max_err


def find_minimal_rank_by_independence():
    """
    Check if the 42 schoolbook rank-1 terms are linearly independent.
    If not, find a minimal subset that still exactly represents the tensor.
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    T = build_target_tensor()
    
    # Build the 42 rank-1 terms as vectors in R^(nm*mp*pn)
    terms = []
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec = np.zeros(nm * mp * pn)
                idx = (i * m + j) * mp * pn + (j * p + k) * pn + (k * n + i)
                vec[idx] = 1.0
                terms.append(vec)
                indices.append((i, j, k))
    
    terms_arr = np.array(terms)  # shape (42, nm*mp*pn)
    T_flat = T.flatten()
    
    # Check: is T in the span of the terms? (It should be, by construction)
    # Find the rank of the term matrix
    rank = np.linalg.matrix_rank(terms_arr.T, tol=1e-10)
    
    # If rank < 42, there are linear dependencies
    # Try to find a minimal subset that spans T
    if rank < len(terms):
        # Use greedy selection: pick terms that maximize span coverage
        selected = []
        span_vectors = []
        
        for idx in range(len(terms)):
            if len(span_vectors) == 0:
                span_vectors = [terms_arr[idx].copy()]
                selected.append(idx)
            else:
                # Check if this term adds to the span
                current_span = np.array(span_vectors)
                new_span = np.vstack([current_span, terms_arr[idx]])
                if np.linalg.matrix_rank(new_span, tol=1e-10) > np.linalg.matrix_rank(current_span, tol=1e-10):
                    span_vectors.append(terms_arr[idx].copy())
                    selected.append(idx)
            
            if len(selected) == rank:
                break
        
        # Verify that the selected terms can represent T
        # Solve: sum_{i in selected} c_i * terms[i] = T
        selected_terms = terms_arr[selected]  # shape (r, nm*mp*pn)
        
        # Least squares solve
        coeffs, residuals, rank2, sv = np.linalg.lstsq(selected_terms.T, T_flat, rcond=None)
        
        if np.allclose(selected_terms.T @ coeffs, T_flat, atol=1e-9):
            # Build the decomposition
            r = len(selected)
            U = [[0] * r for _ in range(nm)]
            V = [[0] * r for _ in range(mp)]
            W = [[0] * r for _ in range(pn)]
            
            for t, idx in enumerate(selected):
                i, j, k = indices[idx]
                c = coeffs[t]
                U[i * m + j][t] = c
                V[j * p + k][t] = 1.0
                W[k * n + i][t] = 1.0
            
            # Verify
            valid, err = verify_decomposition(U, V, W, T)
            if valid:
                return {"U": U, "V": V, "W": W, "rank": r}
    
    return None


def construct():
    """
    Construct a tensor decomposition for 2x3 x 3x7 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    T = build_target_tensor()
    
    # First, check if we can find a lower-rank decomposition via linear independence
    if remaining() > 10:
        result = find_minimal_rank_by_independence()
        if result is not None:
            valid, err = verify_decomposition(result["U"], result["V"], result["W"], T)
            if valid:
                return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fallback to schoolbook decomposition
    return schoolbook_decomposition()
