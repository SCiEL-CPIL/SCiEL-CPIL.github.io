"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _rank1_decomposition(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m * m * p."""
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
    return U, V, W

def _decompose_tensor_to_rank1(U, V, W):
    """
    Convert matrix decomposition back to rank-1 tensor terms.
    Each column c gives a rank-1 term with support on U[:,c], V[:,c], W[:,c].
    """
    n_rows_U = len(U)
    m_rows_V = len(V)
    p_rows_W = len(W)
    rank = len(U[0]) if U else 0
    
    terms = []
    for c in range(rank):
        u_nonzero = [(i, U[i][c]) for i in range(n_rows_U) if U[i][c] != 0]
        v_nonzero = [(j, V[j][c]) for j in range(m_rows_V) if V[j][c] != 0]
        w_nonzero = [(k, W[k][c]) for k in range(p_rows_W) if W[k][c] != 0]
        if u_nonzero and v_nonzero and w_nonzero:
            terms.append((u_nonzero, v_nonzero, w_nonzero))
    return terms

def _term_support(term):
    """Get the set of (U_idx, V_idx, W_idx) tuples with nonzero values in a term."""
    u_set = set(u[0] for u in term[0])
    v_set = set(v[0] for v in term[1])
    w_set = set(w[0] for w in term[2])
    return u_set, v_set, w_set

def _check_term_redundant(term, other_terms, n_rows_U, m_rows_V, p_rows_W):
    """
    Check if a term can be expressed as a linear combination of other terms.
    Returns True if redundant, False otherwise.
    """
    # Build the system: for each (i,j,k) where term has nonzero value,
    # we need sum_c alpha_c * (U_c[i] * V_c[j] * W_c[k]) = U_term[i] * V_term[j] * W_term[k]
    
    # Collect all (i,j,k) positions where term has nonzero product
    u_dict = {u[0]: u[1] for u in term[0]}
    v_dict = {v[0]: v[1] for v in term[1]}
    w_dict = {w[0]: w[1] for w in term[2]}
    
    target_positions = []
    for i, ui in u_dict.items():
        for j, vj in v_dict.items():
            for k, wk in w_dict.items():
                target_value = ui * vj * wk
                target_positions.append((i, j, k, target_value))
    
    if not target_positions:
        return True
    
    # Build matrix A where A[row][col] = U_other[col][i] * V_other[col][j] * W_other[col][k]
    # and b[row] = target_value
    A_rows = []
    b_vec = []
    for (i, j, k, target_val) in target_positions:
        row = []
        for other_term in other_terms:
            ou_dict = {u[0]: u[1] for u in other_term[0]}
            ov_dict = {v[0]: v[1] for v in other_term[1]}
            ow_dict = {w[0]: w[1] for w in other_term[2]}
            if i in ou_dict and j in ov_dict and k in ow_dict:
                row.append(ou_dict[i] * ov_dict[j] * ow_dict[k])
            else:
                row.append(0)
        A_rows.append(row)
        b_vec.append(target_val)
    
    A = np.array(A_rows, dtype=float)
    b = np.array(b_vec, dtype=float)
    
    # Check if b is in the column space of A
    try:
        # Use least squares and check residual
        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        if len(residuals) > 0:
            residual = residuals[0]
        else:
            residual = np.linalg.norm(A @ coeffs - b)
        
        # Check if residual is close to zero relative to norm of b
        b_norm = np.linalg.norm(b)
        if b_norm == 0:
            return True
        return residual < 1e-8 * b_norm
    except Exception:
        return False

def _try_reduce_rank(U, V, W, n, m, p, deadline, max_iterations=500):
    """
    Try to reduce rank by eliminating redundant terms.
    Returns (U, V, W) with potentially reduced rank.
    """
    rank = len(U[0]) if U else 0
    current_rank = rank
    
    # Convert to terms
    terms = _decompose_tensor_to_rank1(U, V, W)
    
    # Try to eliminate terms one by one
    eliminated = set()
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline:
            break
        
        found_elimination = False
        for idx in range(len(terms)):
            if idx in eliminated:
                continue
            if time.time() > deadline:
                break
            
            # Check if this term is redundant given all other non-eliminated terms
            other_terms = [terms[j] for j in range(len(terms)) if j != idx and j not in eliminated]
            
            if _check_term_redundant(terms[idx], other_terms, n*m, m*p, p*n):
                eliminated.add(idx)
                found_elimination = True
                break
        
        if not found_elimination:
            break
        
        iteration += 1
    
    # Reconstruct U, V, W with only non-eliminated terms
    remaining_indices = [i for i in range(len(terms)) if i not in eliminated]
    new_rank = len(remaining_indices)
    
    if new_rank >= current_rank:
        return U, V
    
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    for new_col, old_idx in enumerate(remaining_indices):
        for (i, val) in terms[old_idx][0]:
            new_U[i][new_col] = val
        for (j, val) in terms[old_idx][1]:
            new_V[j][new_col] = val
        for (k, val) in terms[old_idx][2]:
            new_W[k][new_col] = val
    
    return new_U, new_V, new_W

def _construct_2x3x9():
    """
    Construct rank-minimal decomposition for 2x3 * 3x9.
    Starts from schoolbook and attempts rank reduction.
    """
    n, m, p = 2, 3, 9
    start = time.time()
    deadline = start + 130  # Safety margin
    
    # Start with schoolbook
    U, V, W = _rank1_decomposition(n, m, p)
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    Starts from schoolbook rank 54 and attempts to reduce.
    """
    start = time.time()
    deadline = start + 135  # Safety margin before 150s kill
    
    try:
        result = _construct_2x3x9()
        return result
    except Exception as e:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
