"""Matrix multiplication 2x2x10: rank-minimal exact tensor decomposition.

Strategy: Start with rank-35 Strassen baseline, then attempt rank-34 via
alternating minimization on the trilinear tensor equation.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = 140.0  # Safety margin


def _build_strassen_35():
    """Return the known rank-35 Strassen decomposition as numpy arrays."""
    n, m, p = 2, 2, 10
    rank = 35
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            U[:, col] = a_vec
            V[0 * p + 2 * k, col] = b_vec[0]
            V[0 * p + 2 * k + 1, col] = b_vec[1]
            V[1 * p + 2 * k, col] = b_vec[2]
            V[1 * p + 2 * k + 1, col] = b_vec[3]
            W[2 * k * n + 0, col] = c_vec[0]
            W[2 * k * n + 1, col] = c_vec[1]
            W[(2 * k + 1) * n + 0, col] = c_vec[2]
            W[(2 * k + 1) * n + 1, col] = c_vec[3]
            col += 1
    return U, V, W


def _build_target_tensor(n=2, m=2, p=10):
    """Build the target tensor T as a list of (i, j, k, value) for non-zero entries."""
    entries = []
    for i_row in range(n):
        for i_col in range(m):
            i = i_row * m + i_col
            for j_row in range(m):
                for j_col in range(p):
                    j = j_row * p + j_col
                    for k_row in range(n):
                        for k_col in range(p):
                            k = k_col * n + k_row
                            if i_row == k_row and i_col == j_row:
                                entries.append((i, j, k, 1.0))
    return entries


def _residual(U, V, W, target_entries):
    """Compute the residual of the tensor equation."""
    # For each (i,j,k) in target: sum_r U[i,r] V[j,r] W[k,r] should equal 1
    # For all other (i,j,k): sum should equal 0
    # We only check target entries for efficiency
    residual = 0.0
    for i, j, k, val in target_entries:
        s = np.dot(U[i, :], V[j, :] * W[k, :])
        residual += (s - val) ** 2
    return residual


def _solve_for_U(V, W, target_entries, n=2, m=2, p=10):
    """Solve for U given V and W."""
    # For each i: U[i,:] should satisfy sum_r U[i,r] * (V[:,r] * W[:,r]) = T[i,j,k] for all (j,k)
    # This is a linear system: U[i,:] * M_i = b_i where M_i[j,k] = V[j,r]*W[k,r] summed over r
    # Actually, for fixed i, the equation is: for each (j,k), sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
    # Let M[i,j,k] = sum_r V[j,r] W[k,r] ... no, that's not right.
    # The equation is trilinear. Fixing V and W, it's linear in U.
    # For fixed i, we have: sum_r U[i,r] * (V[:,r] * W[:,r])[j,k] = T[i,j,k]
    # But V[:,r] * W[:,r] is not a matrix product in the usual sense.
    # Actually: for fixed r, V[:,r] is a vector of length m*p=20, W[:,r] is a vector of length p*n=20.
    # The product V[j,r] * W[k,r] for all j,k gives a 20x20 matrix.
    # So for fixed i: sum_r U[i,r] * (V[:,r] outer W[:,r])[j,k] = T[i,j,k]
    # This is: U[i,:] dot (vectorized sum over r of V[:,r] outer W[:,r]) = T[i,:]
    # Let's build the system: for each i, we have (m*p)*(p*n) = 400 equations in R unknowns.
    # But T[i,j,k] is zero for most (j,k), so we only need to satisfy the non-zero ones.
    
    R = V.shape[1]
    U = np.zeros((n * m, R))
    
    for i in range(n * m):
        # Build linear system: A @ U[i,:] = b
        # A has one row per (j,k) where T[i,j,k] != 0
        # A[row, r] = V[j,r] * W[k,r]
        # b[row] = T[i,j,k]
        rows = []
        b = []
        for j in range(m * p):
            for k in range(p * n):
                # Check if T[i,j,k] should be 1
                i_row = i // m
                i_col = i % m
                j_row = j // p
                j_col = j % p
                k_row = k % n
                k_col = k // n
                if i_row == k_row and i_col == j_row:
                    rows.append(V[j, :] * W[k, :])
                    b.append(1.0)
        
        if not rows:
            continue
        
        A = np.array(rows)  # shape (num_eq, R)
        b = np.array(b)
        
        # Solve least squares
        if A.shape[0] >= R:
            U[i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        else:
            # Underdetermined, use minimum norm
            U[i, :] = A.T @ np.linalg.lstsq(A, A.T, rcond=None)[0]
            # Actually, just use lstsq which handles both cases
            U[i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return U


def _solve_for_V(U, W, target_entries, n=2, m=2, p=10):
    """Solve for V given U and W."""
    R = U.shape[1]
    V = np.zeros((m * p, R))
    
    for j in range(m * p):
        rows = []
        b = []
        for i in range(n * m):
            for k in range(p * n):
                i_row = i // m
                i_col = i % m
                j_row = j // p
                j_col = j % p
                k_row = k % n
                k_col = k // n
                if i_row == k_row and i_col == j_row:
                    rows.append(U[i, :] * W[k, :])
                    b.append(1.0)
        
        if not rows:
            continue
        
        A = np.array(rows)
        b = np.array(b)
        V[j, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return V


def _solve_for_W(U, V, target_entries, n=2, m=2, p=10):
    """Solve for W given U and V."""
    R = U.shape[1]
    W = np.zeros((p * n, R))
    
    for k in range(p * n):
        rows = []
        b = []
        for i in range(n * m):
            for j in range(m * p):
                i_row = i // m
                i_col = i % m
                j_row = j // p
                j_col = j % p
                k_row = k % n
                k_col = k // n
                if i_row == k_row and i_col == j_row:
                    rows.append(U[i, :] * V[j, :])
                    b.append(1.0)
        
        if not rows:
            continue
        
        A = np.array(rows)
        b = np.array(b)
        W[k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return W


def _try_rank_34(start_time):
    """Try to find a rank-34 decomposition via alternating minimization."""
    n, m, p = 2, 2, 10
    target_entries = _build_target_tensor(n, m, p)
    
    # Initialize from Strassen-35 by dropping one term
    U35, V35, W35 = _build_strassen_35()
    
    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try dropping each term and running alternating minimization
    for drop_idx in range(35):
        if time.time() - start_time > _DEADLINE - 30:
            break
        
        U = U35[:, [r for r in range(35) if r != drop_idx]]
        V = V35[:, [r for r in range(35) if r != drop_idx]]
        W = W35[:, [r for r in range(35) if r != drop_idx]]
        
        # Alternating minimization
        for iteration in range(50):
            if time.time() - start_time > _DEADLINE - 30:
                break
            
            U = _solve_for_U(V, W, target_entries, n, m, p)
            V = _solve_for_V(U, W, target_entries, n, m, p)
            W = _solve_for_W(U, V, target_entries, n, m, p)
        
        # Check residual
        res = _residual(U, V, W, target_entries)
        if res < best_residual:
            best_residual = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < 1e-6:
            break
    
    if best_residual < 1e-6:
        # Verify with random matrices
        rng = np.random.default_rng(999)
        A_test = rng.integers(-3, 4, size=(n, m)).astype(float)
        B_test = rng.integers(-3, 4, size=(m, p)).astype(float)
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        C_computed = np.zeros(p * n)
        for r in range(34):
            uA = np.dot(best_U[:, r], A_flat)
            vB = np.dot(best_V[:, r], B_flat)
            C_computed += uA * vB * best_W[:, r]
        
        C_expected_reordered = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_expected_reordered[k * n + i] = C_expected[i, k]
        
        if np.allclose(C_computed, C_expected_reordered, atol=1e-6):
            return best_U, best_V, best_W
    
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2x10 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' of shapes (4, rank), (20, rank), (20, rank).
    """
    start_time = time.time()
    n, m, p = 2, 2, 10
    
    # Start with rank-35 Strassen
    U35, V35, W35 = _build_strassen_35()
    
    # Try to find rank-34
    result_34 = _try_rank_34(start_time)
    
    if result_34 is not None:
        U, V, W = result_34
    else:
        U, V, W = U35, V35, W35
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
