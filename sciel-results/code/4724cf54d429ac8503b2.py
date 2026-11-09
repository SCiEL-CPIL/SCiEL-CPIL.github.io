"""
Matrix multiplication tensor rank minimization for 2x4 * 4x7.
Uses exact linear-algebra-based rank reduction exploiting the bilinear
structure of the matrix multiplication tensor.
"""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 4, 7]


def _verify(U, V, W, tol=1e-9):
    """Check whether the decomposition exactly reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i]))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _objective(U, V, W, n, m, p):
    """Sum of squared errors over all (i,j,k) pairs."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            w_rows = W.reshape(p, n, -1)[:, i, :]
            v_rows = V.reshape(m, p, -1)[j, :, :]
            vw = v_rows * w_rows
            t_vals = u_row @ vw.T
            err += float(np.sum((t_vals - 1.0) ** 2))
    return err


def _build_constraint_matrix(U, V, n, m, p, rank):
    """
    Build the linear constraint matrix for W.
    For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    Group by W row index (k*n+i). For a fixed (k,i), the W row W[k*n+i,:] 
    appears in constraints for all j.
    
    The constraint for (i,j,k) is:
    W[k*n+i,:] dot (U[i*m+j,:] * V[j*p+k,:]) = 1
    
    So for each distinct W row index w_idx = k*n+i, we have m constraints:
    W[w_idx,:] dot (U[i*m+j,:] * V[j*p+k,:]) = 1, for j=0..m-1
    
    This is a system: A_w * W[w_idx,:]^T = b_w where A_w has m rows.
    """
    # For each (k, i), collect the m constraint vectors
    # W row index = k * n + i
    # Constraint vectors: for each j, vec_j = U[i*m+j,:] * V[j*p+k,:]
    
    # Build full system: for each (i,j,k), we have one equation
    # But W[k*n+i,c] appears in equations for all j
    # 
    # Let's build it differently: the full system is linear in all W entries.
    # Number of unknowns: p*n*rank = 14*rank
    # Number of equations: n*m*p = 56
    #
    # Equation for (i,j,k): sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1
    # 
    # This is a linear system in W entries.
    # Let's vectorize: W_flat is (p*n*rank,)
    # For equation (i,j,k): the coefficient of W_flat[k*n*rank + c] is U[i*m+j,c]*V[j*p+k,c]
    
    A = np.zeros((n * m * p, p * n * rank))
    b = np.ones(n * m * p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for c in range(rank):
                    w_flat_idx = (k * n + i) * rank + c
                    A[eq_idx, w_flat_idx] = U[i * m + j, c] * V[j * p + k, c]
    
    return A, b


def _build_constraint_matrix_U(V, W, n, m, p, rank):
    """Build linear constraint matrix for U given fixed V, W."""
    # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Unknowns: U entries, n*m*rank = 8*rank
    A = np.zeros((n * m * p, n * m * rank))
    b = np.ones(n * m * p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for c in range(rank):
                    u_flat_idx = (i * m + j) * rank + c
                    A[eq_idx, u_flat_idx] = V[j * p + k, c] * W[k * n + i, c]
    
    return A, b


def _build_constraint_matrix_V(U, W, n, m, p, rank):
    """Build linear constraint matrix for V given fixed U, W."""
    # Equation for (i,j,k): sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1
    # Unknowns: V entries, m*p*rank = 28*rank
    A = np.zeros((n * m * p, m * p * rank))
    b = np.ones(n * m * p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for c in range(rank):
                    v_flat_idx = (j * p + k) * rank + c
                    A[eq_idx, v_flat_idx] = U[i * m + j, c] * W[k * n + i, c]
    
    return A, b


def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank from current to target_rank using linear algebra.
    
    Strategy: 
    1. Start with current decomposition of rank R.
    2. Try to find a lower-rank decomposition by:
       a. Fixing a subset of columns and solving for the rest
       b. Using the fact that the system is linear in each factor
    
    Approach: Use alternating linear solves.
    - Fix U and V, solve for W (linear system)
    - Fix V and W, solve for U (linear system)  
    - Fix U and W, solve for V (linear system)
    - Check if we can eliminate columns (set some to zero) while maintaining feasibility
    """
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W, _objective(U, V, W, n, m, p)
    
    # Strategy: try to find a decomposition of exactly target_rank
    # by randomly projecting and solving
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _objective(U, V, W, n, m, p)
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Randomly select target_rank columns to keep, try to solve
        # This is like a random projection
        
        # Initialize with random rank-target_rank matrices
        rng = np.random.RandomState(attempt * 31 + 7)
        
        U_new = rng.randn(n * m, target_rank) * 0.5
        V_new = rng.randn(m * p, target_rank) * 0.5
        W_new = rng.randn(p * n, target_rank) * 0.5
        
        # Alternating linear solves
        for iteration in range(30):
            if time.time() > deadline - 3:
                break
            
            # Solve for W given U_new, V_new
            A_W, b_W = _build_constraint_matrix(U_new, V_new, n, m, p, target_rank)
            try:
                W_flat, residuals, rank_W, sv = linalg.lstsq(A_W, b_W)
                W_new = W_flat.reshape(p * n, target_rank)
            except:
                break
            
            # Solve for U given V_new, W_new
            A_U, b_U = _build_constraint_matrix_V(V_new, W_new, n, m, p, target_rank)
            try:
                U_flat, residuals, rank_U, sv = linalg.lstsq(A_U, b_U)
                U_new = U_flat.reshape(n * m, target_rank)
            except:
                break
            
            # Solve for V given U_new, W_new
            A_V, b_V = _build_constraint_matrix_U(U_new, W_new, n, m, p, target_rank)
            try:
                V_flat, residuals, rank_V, sv = linalg.lstsq(A_V, b_V)
                V_new = V_flat.reshape(m * p, target_rank)
            except:
                break
            
            err = _objective(U_new, V_new, W_new, n, m, p)
            if err < best_err:
                best_err = err
                best_U = U_new.copy()
                best_V = V_new.copy()
                best_W = W_new.copy()
            
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    return best_U, best_V, best_W, best_err


def _exact_polish(U, V, W, n, m, p, max_iters=50, tol=1e-12):
    """
    Exact polish using alternating linear solves.
    Each step solves a linear system exactly (up to numerical precision).
    """
    rank = U.shape[1]
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    for it in range(max_iters):
        # Solve for W
        A_W, b_W = _build_constraint_matrix(U, V, n, m, p, rank)
        W_flat, _, _, _ = linalg.lstsq(A_W, b_W)
        W = W_flat.reshape(p * n, rank)
        
        # Solve for U
        A_U, b_U = _build_constraint_matrix_V(V, W, n, m, p, rank)
        U_flat, _, _, _ = linalg.lstsq(A_U, b_U)
        U = U_flat.reshape(n * m, rank)
        
        # Solve for V
        A_V, b_V = _build_constraint_matrix_U(U, W, n, m, p, rank)
        V_flat, _, _, _ = linalg.lstsq(A_V, b_V)
        V = V_flat.reshape(m * p, rank)
        
        err = _objective(U, V, W, n, m, p)
        if err < tol:
            break
    
    return U, V, W, err


def _to_half_int_matrix(M, tol=1e-8):
    """Check if all entries are within tol of half-integers; return rounded version."""
    M = np.array(M, dtype=float)
    scaled = M * 2.0
    rounded = np.round(scaled)
    if np.max(np.abs(scaled - rounded)) > tol:
        return None
    return (rounded / 2.0)


def _snip_columns(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to eliminate columns from the current decomposition to reach target_rank.
    For each column c, check if setting it to zero still allows a solution.
    """
    current_rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Try removing columns one at a time
    for c in range(current_rank - 1, target_rank - 1, -1):
        if time.time() > deadline - 5:
            break
        
        # Try setting column c to zero and re-solving
        U_trial = U.copy()
        V_trial = V.copy()
        W_trial = W.copy()
        U_trial[:, c] = 0
        V_trial[:, c] = 0
        W_trial[:, c] = 0
        
        # Polish with remaining columns
        U_t, V_t, W_t, err = _exact_polish(U_trial, V_trial, W_trial, n, m, p, max_iters=20, tol=1e-10)
        
        if err < 1e-8:
            # Successfully eliminated column c
            U = U_t[:, :c]  # Remove column c
            V = V_t[:, :c]
            W = W_t[:, :c]
            if U.shape[1] <= target_rank:
                break
    
    return U, V, W, _objective(U, V, W, n, m, p)


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 56
    deadline = time.time() + 140  # safety margin
    
    # Start with schoolbook as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_rank = schoolbook_rank
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Try to reduce rank from 56 down toward 28
    
    # Phase 1: Try direct rank reduction from schoolbook
    for target_rank in range(schoolbook_rank - 1, 27, -1):
        if time.time() > deadline - 20:
            break
        
        U_cur, V_cur, W_cur = U_sb.copy(), V_sb.copy(), W_sb.copy()
        
        # Try to reduce to target_rank
        U_new, V_new, W_new, err = _try_rank_reduction(
            U_cur, V_cur, W_cur, n, m, p, target_rank, deadline
        )
        
        if err < 1e-8:
            # Verify
            Uh = _to_half_int_matrix(U_new, tol=1e-6)
            Vh = _to_half_int_matrix(V_new, tol=1e-6)
            Wh = _to_half_int_matrix(W_new, tol=1e-6)
            
            if Uh is not None and Vh is not None and Wh is not None:
                if _verify(Uh.tolist(), Vh.tolist(), Wh.tolist()):
                    best_rank = target_rank
                    best_U = Uh
                    best_V = Vh
                    best_W = Wh
                    break
            elif _verify(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                best_rank = target_rank
                best_U = U_new
                best_V = V_new
                best_W = W_new
                break
        
        # Also try column snipping from current best
        if best_rank > target_rank:
            U_snip, V_snip, W_snip, err_snip = _snip_columns(
                best_U, best_V, best_W, n, m, p, target_rank, deadline
            )
            if err_snip < 1e-8 and U_snip.shape[1] < best_rank:
                Uh = _to_half_int_matrix(U_snip, tol=1e-6)
                Vh = _to_half_int_matrix(V_snip, tol=1e-6)
                Wh = _to_half_int_matrix(W_snip, tol=1e-6)
                if Uh is not None and Vh is not None and Wh is not None:
                    if _verify(Uh.tolist(), Vh.tolist(), Wh.tolist()):
                        best_rank = U_snip.shape[1]
                        best_U = Uh
                        best_V = Vh
                        best_W = Wh
                        break
                elif _verify(U_snip.tolist(), V_snip.tolist(), W_snip.tolist()):
                    best_rank = U_snip.shape[1]
                    best_U = U_snip
                    best_V = V_snip
                    best_W = W_snip
                    break
    
    # Phase 2: If we found something, try to polish further
    if best_rank < schoolbook_rank and time.time() < deadline - 10:
        U_pol, V_pol, W_pol, err_pol = _exact_polish(
            best_U, best_V, best_W, n, m, p, max_iters=50, tol=1e-12
        )
        if err_pol < 1e-10:
            best_U = U_pol
            best_V = V_pol
            best_W = W_pol
    
    # Convert to list format
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
        best_V = best_V.tolist()
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}


# EVOLVE-BLOCK-END
