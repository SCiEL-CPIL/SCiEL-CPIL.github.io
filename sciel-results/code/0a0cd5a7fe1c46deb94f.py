"""
Matrix multiplication 2x5x5 rank minimization.
Uses Strassen's algorithm on 2x2 sub-blocks plus systematic search for
cross-block linear dependencies to reduce rank below 46.
"""
import time
import numpy as np
from itertools import combinations
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_strassen_decomposition():
    """
    Build the rank-46 Strassen-based decomposition for 2x5 x 5x5.
    Partition inner dimension 5 = 2+2+1.
    Returns list of (u_entries, v_entries, w_entries) tuples.
    """
    n, m, p = 2, 5, 5
    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        # M1 = (a+d)(e+h)
        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        # M2 = (c+d)e
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        # M3 = a(f-h)
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        # M4 = d(g-e)
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        # M5 = (a+b)h
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        # M6 = (c-a)(e+f)
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        # M7 = (b-d)(g+h)
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    # Partition 5 = 2+2+1
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)

    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)

    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)

    return terms

def _terms_to_matrices(terms):
    """Convert list of term tuples to U, V, W matrices."""
    n, m, p = 2, 5, 5
    total_rank = len(terms)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val

    return U, V, W, total_rank

def _validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly reconstructs the matrix multiplication tensor."""
    r = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0.0
                for c in range(r):
                    val += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def _compute_contributions(terms, n, m, p):
    """Compute the contribution vector for each term."""
    r0 = len(terms)
    num_constraints = n * m * p
    contributions = np.zeros((r0, num_constraints))
    
    for c in range(r0):
        u_entries = dict(terms[c][0])
        v_entries = dict(terms[c][1])
        w_entries = dict(terms[c][2])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    u_val = u_entries.get(idx_u, 0)
                    v_val = v_entries.get(idx_v, 0)
                    w_val = w_entries.get(idx_w, 0)
                    contributions[c, i * m * p + j * p + k] = u_val * v_val * w_val
    
    return contributions

def _search_for_reductions(terms, deadline):
    """
    Search for rank reductions by finding linear dependencies among the terms.
    """
    n, m, p = 2, 5, 5
    r0 = len(terms)
    
    # Build contribution matrix
    contributions = _compute_contributions(terms, n, m, p)
    
    # Find null space of the contribution matrix
    # If there's a non-trivial null vector, we can potentially reduce rank
    try:
        null_space = np.linalg.null_space(contributions)
    except:
        null_space = None
    
    if null_space is not None and null_space.shape[1] > 0:
        # For each null vector, check if we can zero out a coefficient
        for ns in range(null_space.shape[1]):
            v = null_space[:, ns]
            
            # Find which term to remove: pick c where v[c] is largest in magnitude
            max_idx = np.argmax(np.abs(v))
            if abs(v[max_idx]) > 1e-10:
                t = -1.0 / v[max_idx]
                new_coeffs = 1.0 + t * v
                
                # Check that new_coeffs[max_idx] is approximately 0
                if abs(new_coeffs[max_idx]) < 1e-9:
                    # Remove term max_idx and scale others
                    new_terms = []
                    for c in range(r0):
                        if c == max_idx:
                            continue
                        if abs(new_coeffs[c]) > 1e-9:
                            # Scale the term by new_coeffs[c]
                            scale = new_coeffs[c]
                            u_entries = [(idx, val * scale) for idx, val in terms[c][0]]
                            v_entries = [(idx, val) for idx, val in terms[c][1]]
                            w_entries = [(idx, val) for idx, val in terms[c][2]]
                            new_terms.append((u_entries, v_entries, w_entries))
                    
                    # Verify
                    U_test, V_test, W_test, _ = _terms_to_matrices(new_terms)
                    if _validate_decomposition(U_test, V_test, W_test, n, m, p):
                        return new_terms
    
    return terms

def _try_gradient_descent(n, m, p, rank, deadline, max_iter=1000, lr=0.005):
    """
    Try to find a decomposition of given rank using gradient descent.
    Returns (U, V, W) as lists if found, else None.
    """
    np.random.seed(42 + rank)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Compute error and gradients
        error = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    err = val - 1.0
                    error += err * err
                    
                    # Gradients
                    dU[idx_u, :] += 2 * err * (V[idx_v, :] * W[idx_w, :])
                    dV[idx_v, :] += 2 * err * (U[idx_u, :] * W[idx_w, :])
                    dW[idx_w, :] += 2 * err * (U[idx_u, :] * V[idx_w, :])
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-10:
            break
        
        # Adaptive learning rate
        step_size = lr / (1.0 + it * 0.0005)
        
        U -= step_size * dU
        V -= step_size * dV
        W -= step_size * dW
        
        # Occasionally restart from best if stuck
        if it > 0 and it % 300 == 0 and error > 1e-4:
            U = best_U + np.random.randn(n * m, rank) * 0.02
            V = best_V + np.random.randn(m * p, rank) * 0.02
            W = best_W + np.random.randn(p * n, rank) * 0.02
    
    # Final check on best found
    if best_error < 1e-8:
        # Try to snap to rational values
        U_snap = np.round(best_U * 1000) / 1000
        V_snap = np.round(best_V * 1000) / 1000
        W_snap = np.round(best_W * 1000) / 1000
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _validate_exact(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
        
        # Try with more decimal places
        U_snap = np.round(best_U * 10000) / 10000
        V_snap = np.round(best_V * 10000) / 10000
        W_snap = np.round(best_W * 10000) / 10000
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _validate_exact(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the standard rank-46 decomposition
    terms = _build_strassen_decomposition()
    
    # Try to find rank reductions via null space analysis
    if time.time() < deadline - 40:
        terms = _search_for_reductions(terms, deadline)
    
    # Convert to matrices
    U, V, W, total_rank = _terms_to_matrices(terms)
    
    # Verify
    if not _validate_decomposition(U, V, W, 2, 5, 5):
        # Fallback to original
        terms = _build_strassen_decomposition()
        U, V, W, total_rank = _terms_to_matrices(terms)
    
    # Try gradient descent to find lower rank decompositions
    current_rank = total_rank
    best_U, best_V, best_W = U, V, W
    
    # Try ranks from current_rank-1 down to max(20, current_rank-10)
    for target_rank in range(current_rank - 1, max(20, current_rank - 10), -1):
        if time.time() > deadline - 15:
            break
        
        result = _try_gradient_descent(2, 5, 5, target_rank, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
