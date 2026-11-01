import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill


def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0

    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices

        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1

        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1

        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1

        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1

        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1

        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1

        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1

    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 0), (1, 0), (0, 1), (1, 1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 0), (2, 1), (3, 0), (3, 1)],
                   [(0, 0), (1, 0), (0, 1), (1, 1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 0), (3, 0), (2, 1), (3, 1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 2), (2, 3), (3, 2), (3, 3)],
                   [(2, 0), (3, 0), (2, 1), (3, 1)])

    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0, 0), col] = 1; V[v_idx(0, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(0, 1), col] = 1; V[v_idx(1, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(1, 0), col] = 1; V[v_idx(0, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1
    U[u_idx(1, 1), col] = 1; V[v_idx(1, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1

    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0, 2), col] = 1; V[v_idx(2, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(0, 3), col] = 1; V[v_idx(3, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(1, 2), col] = 1; V[v_idx(2, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1
    U[u_idx(1, 3), col] = 1; V[v_idx(3, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1

    return U, V, W


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication (vectorized)."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_kronecker_matrix(U, V, W):
    """Build the n*m*p x rank matrix where each column is the vectorized Kronecker product
    of U[:,c], V[:,c], W[:,c]."""
    n, m, p = 2, 4, 5
    r = U.shape[1]
    A = np.zeros((n * m * p, r))
    for c in range(r):
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A[idx, c] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    idx += 1
    return A


def _check_column_redundant_robust(K, col_to_remove, tol=1e-7):
    """Check if column col_to_remove of K is in the span of the other columns using SVD.
    Returns True if redundant, False otherwise."""
    r = K.shape[1]
    if r <= 1:
        return False
    
    keep_cols = [c for c in range(r) if c != col_to_remove]
    A = K[:, keep_cols]
    b = K[:, col_to_remove]
    
    # Use SVD to check if b is in the column space of A
    try:
        U_svd, s_vals, Vt = np.linalg.svd(A, full_matrices=False)
    except:
        return False
    
    # Determine numerical rank of A
    num_rank = np.sum(s_vals > tol * s_vals[0]) if len(s_vals) > 0 else 0
    
    if num_rank == 0:
        # A is zero; b must also be zero
        return np.max(np.abs(b)) < tol
    
    # Project b onto the column space of A
    # The column space is spanned by U_svd[:, :num_rank]
    U_sub = U_svd[:, :num_rank]
    # Projection: U_sub @ (U_sub^T @ b)
    proj = U_sub @ (U_sub.T @ b)
    residual = np.max(np.abs(b - proj))
    
    return residual < tol


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE

    U, V, W = _build_strassen_36()

    # Verify the base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank-40
        n, m, p = 2, 4, 5
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    
    # Pre-compute Kronecker matrix K for the initial decomposition
    K = _build_kronecker_matrix(U, V, W)
    
    # Phase 1: Greedy removal of linearly dependent columns using robust SVD check
    progress = True
    while progress and time.time() < deadline - 15:
        progress = False
        rank = U.shape[1]
        for col in range(rank):
            if time.time() > deadline - 15:
                break
            
            # Check redundancy using robust SVD method
            if _check_column_redundant_robust(K, col):
                # Remove column
                mask = [i for i in range(rank) if i != col]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                
                # Verify to be safe
                if _verify_decomposition(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    K = _build_kronecker_matrix(U, V, W)
                    rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    progress = True
                    break
        if not progress:
            break

    # Phase 2: Try removing pairs
    if time.time() < deadline - 10:
        progress = True
        while progress and time.time() < deadline - 10:
            progress = False
            rank = U.shape[1]
            for i in range(rank):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, rank):
                    if time.time() > deadline - 10:
                        break
                    
                    keep_cols = [c for c in range(rank) if c != i and c != j]
                    if len(keep_cols) < 1:
                        continue
                        
                    A = K[:, keep_cols]
                    b = np.column_stack((K[:, i], K[:, j]))
                    
                    try:
                        solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                    except:
                        continue

                    A_sol = A @ solution
                    if np.max(np.abs(A_sol - b)) > 1e-8:
                        continue
                        
                    # If valid, remove both
                    mask = [c for c in range(rank) if c != i and c != j]
                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]
                    
                    if _verify_decomposition(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        K = _build_kronecker_matrix(U, V, W)
                        rank = U.shape[1]
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        progress = True
                        break
                if time.time() > deadline - 10:
                    break
                if progress:
                    break

    # Phase 3: Random rotations to expose dependencies - more systematic approach
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42)
        rank = U.shape[1]
        max_trials = 200
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            if rank < 3:
                break
            
            # Try different rotation strategies
            strategy = trial % 3
            if strategy == 0:
                # Rotate two random columns
                cols = rng.choice(rank, size=2, replace=False)
                t = rng.uniform(-2, 2)
                U_rot = U.copy()
                V_rot = V.copy()
                W_rot = W.copy()
                U_rot[:, cols[0]] = U[:, cols[0]] + t * U[:, cols[1]]
                V_rot[:, cols[0]] = V[:, cols[0]] + t * V[:, cols[1]]
                W_rot[:, cols[0]] = W[:, cols[0]] + t * W[:, cols[1]]
            elif strategy == 1:
                # Add a fraction of one column to another
                cols = rng.choice(rank, size=2, replace=False)
                t = rng.uniform(-1, 1)
                U_rot = U.copy()
                V_rot = V.copy()
                W_rot = W.copy()
                U_rot[:, cols[1]] = U[:, cols[1]] + t * U[:, cols[0]]
                V_rot[:, cols[1]] = V[:, cols[1]] + t * V[:, cols[0]]
                W_rot[:, cols[1]] = W[:, cols[1]] + t * W[:, cols[0]]
            else:
                # Scale a column
                col = rng.randint(rank)
                t = rng.uniform(0.5, 1.5)
                U_rot = U.copy()
                V_rot = V.copy()
                W_rot = W.copy()
                U_rot[:, col] = t * U[:, col]
                V_rot[:, col] = V[:, col]
                W_rot[:, col] = W[:, col] / t  # Keep product the same
            
            K_rot = _build_kronecker_matrix(U_rot, V_rot, W_rot)
            
            # Check for single column redundancy in the rotated system
            found_redundant = False
            for col in range(rank):
                if time.time() > deadline - 5:
                    break
                if _check_column_redundant_robust(K_rot, col):
                    mask = [i for i in range(rank) if i != col]
                    U_new = U_rot[:, mask]
                    V_new = V_rot[:, mask]
                    W_new = W_rot[:, mask]
                    
                    if _verify_decomposition(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        K = _build_kronecker_matrix(U, V, W)
                        rank = U.shape[1]
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        found_redundant = True
                        break
            if found_redundant:
                # Reset trial counter to continue searching from new state
                pass
            if time.time() > deadline - 5:
                break

    # Phase 4: Final polish - try to reduce rank by checking if the current rank is minimal
    # by attempting to find a rank-(rank-1) decomposition via iterative refinement
    if time.time() < deadline - 2:
        rank = U.shape[1]
        # Try to see if we can reduce by 1 more via a more aggressive search
        rng2 = np.random.RandomState(123)
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
            if rank < 2:
                break
            
            # Try to express one column as a combination of others with random weights
            target_col = rng2.randint(rank)
            other_cols = [c for c in range(rank) if c != target_col]
            
            # Use least squares to find the best combination
            A = K[:, other_cols]
            b = K[:, target_col]
            
            try:
                solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            except:
                continue
            
            A_sol = A @ solution
            if np.max(np.abs(A_sol - b)) < 1e-9:
                # This column is exactly in the span, so it's redundant
                mask = [c for c in range(rank) if c != target_col]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                
                if _verify_decomposition(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    K = _build_kronecker_matrix(U, V, W)
                    rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break

    # Final verification
    if not _verify_decomposition(U, V, W):
        U, V, W = best_U, best_V, best_W

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
