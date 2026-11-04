"""
Rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication.

Strategy:
  1. Build a structured rank-44 decomposition (4 Strassen 2x2x2 blocks + 4 schoolbook 1x2x2 blocks).
  2. Attempt to reduce rank by finding linear dependencies among the tensor terms and eliminating them.
  3. Verify each candidate decomposition exactly.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1 (the matrix multiplication tensor)."""
    return np.ones((n, m, p), dtype=np.float64)


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)


def _construct_structured_decomposition():
    """Construct a rank-44 decomposition using Strassen for 2x2x2 blocks and schoolbook for 1x2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Term 1: (A11+A22)(B11+B22)
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 2: (A21+A22)B11
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = -1
        col += 1
        # Term 3: A11(B21-B11)
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = -1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 4: A22(B12-B22)
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = -1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        col += 1
        # Term 5: (A11+A12)B22
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = -1
        col += 1
        # Term 6: (A21-A11)(B11+B12)
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 0), col] = -1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 7: (A12-A22)(B21+B22)
        U[(rA + 0) * m + (cA + 1), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = -1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1

    # Apply Strassen to the four 2x2x2 sub-blocks
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    # Apply schoolbook to the four 1x2x2 sub-blocks (row 2 of A, columns 0-1 or 2-3)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)

    return U, V, W


def _compute_outer_products(U, V, W, n, m, p):
    """Precompute the flattened outer product for each term."""
    rank = U.shape[1]
    outers = np.zeros((n * m * p, rank))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        outers[:, r] = np.einsum('ij,jk,ki->ijk', u, v, w).flatten()
    return outers


def _try_eliminate_one_term(U, V, W, n, m, p):
    """Try to eliminate one term by finding linear dependency in U (or V or W) columns.
    
    If column r of U is a linear combination of other U columns, we can absorb
    the coefficients into V and W to eliminate that term.
    """
    rank = U.shape[1]
    for r_to_remove in range(rank):
        indices = [i for i in range(rank) if i != r_to_remove]
        
        # Check if U[:, r_to_remove] is in span of U[:, indices]
        try:
            # Solve U[:, indices] @ delta = U[:, r_to_remove]
            delta_u, residuals_u, rank_u, sv_u = np.linalg.lstsq(
                U[:, indices], U[:, r_to_remove], rcond=1e-10
            )
            if rank_u < len(indices) and np.allclose(
                U[:, indices] @ delta_u, U[:, r_to_remove], atol=1e-9, rtol=0
            ):
                # U[:, r] = sum_j delta_u[j] * U[:, indices[j]]
                # So term r: U_r V_r W_r = sum_j delta_u[j] U_j V_r W_r
                # We can add delta_u[j] * V_r to V_j and delta_u[j] * W_r to W_j
                U_new = U[:, indices].copy()
                V_new = V[:, indices].copy()
                W_new = W[:, indices].copy()
                for j, orig_idx in enumerate(indices):
                    V_new[:, j] += delta_u[j] * V[:, r_to_remove]
                    W_new[:, j] += delta_u[j] * W[:, r_to_remove]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    return U_new, V_new, W_new
        except Exception:
            pass
        
        # Also try V dependency
        try:
            delta_v, residuals_v, rank_v, sv_v = np.linalg.lstsq(
                V[:, indices], V[:, r_to_remove], rcond=1e-10
            )
            if rank_v < len(indices) and np.allclose(
                V[:, indices] @ delta_v, V[:, r_to_remove], atol=1e-9, rtol=0
            ):
                U_new = U[:, indices].copy()
                V_new = V[:, indices].copy()
                W_new = W[:, indices].copy()
                for j, orig_idx in enumerate(indices):
                    U_new[:, j] += delta_v[j] * U[:, r_to_remove]
                    W_new[:, j] += delta_v[j] * W[:, r_to_remove]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    return U_new, V_new, W_new
        except Exception:
            pass
        
        # Also try W dependency
        try:
            delta_w, residuals_w, rank_w, sv_w = np.linalg.lstsq(
                W[:, indices], W[:, r_to_remove], rcond=1e-10
            )
            if rank_w < len(indices) and np.allclose(
                W[:, indices] @ delta_w, W[:, r_to_remove], atol=1e-9, rtol=0
            ):
                U_new = U[:, indices].copy()
                V_new = V[:, indices].copy()
                W_new = W[:, indices].copy()
                for j, orig_idx in enumerate(indices):
                    U_new[:, j] += delta_w[j] * U[:, r_to_remove]
                    V_new[:, j] += delta_w[j] * V[:, r_to_remove]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    return U_new, V_new, W_new
        except Exception:
            pass

    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_reductions, time_budget):
    """Iteratively try to reduce the rank by eliminating terms."""
    start_time = time.time()
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    current_rank = U_cur.shape[1]

    for _ in range(max_reductions):
        if time.time() - start_time > time_budget:
            break
        U_cur, V_cur, W_cur = _try_eliminate_one_term(U_cur, V_cur, W_cur, n, m, p)
        if U_cur.shape[1] == current_rank:
            # No reduction was made, so stop
            break
        current_rank = U_cur.shape[1]

    return U_cur, V_cur, W_cur


def _try_randomized_reduction(U, V, W, n, m, p, time_budget):
    """Try to reduce rank via random perturbations and rank reduction.
    
    Idea: add small random noise to the decomposition, then use SVD to find
    a lower-rank approximation, then verify if it's exact.
    """
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Precompute outer products for fast verification
    n_mp = n * m * p
    
    for attempt in range(200):
        if time.time() - start_time > time_budget:
            break
        
        # Try different random seeds
        np.random.seed(attempt * 12345 + 67890)
        
        # Create a random perturbation
        eps = 1e-8
        U_pert = U + eps * np.random.randn(*U.shape)
        V_pert = V + eps * np.random.randn(*V.shape)
        W_pert = W + eps * np.random.randn(*W.shape)
        
        # Compute the tensor from perturbed decomposition
        T_pert = np.zeros((n, m, p), dtype=np.float64)
        for r in range(U_pert.shape[1]):
            u = U_pert[:, r].reshape(n, m)
            v = V_pert[:, r].reshape(m, p)
            w = W_pert[:, r].reshape(p, n)
            T_pert += np.einsum('ij,jk,ki->ijk', u, v, w)
        
        # The target tensor is all ones
        T_target = np.ones((n, m, p), dtype=np.float64)
        
        # Check if the perturbation is small
        if not np.allclose(T_pert, T_target, atol=1e-6, rtol=0):
            continue
        
        # Try to find a lower-rank decomposition via SVD of the tensor
        # Flatten the tensor and use matrix decomposition
        T_flat = T_target.flatten()
        
        # This is a heuristic - try to see if we can reduce rank
        # by checking if the current decomposition has redundancy
        # For now, just try the elimination approach on the perturbed version
        U_red, V_red, W_red = _try_eliminate_one_term(U_pert, V_pert, W_pert, n, m, p)
        
        if U_red.shape[1] < best_rank:
            # Verify the reduced decomposition
            if _verify_decomposition(U_red, V_red, W_red, n, m, p, tol=1e-9):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = U_red.shape[1]
    
    return best_U, best_V, best_W


def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = 44

    # Verify the initial decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback: schoolbook rank-48 decomposition
        U48 = np.zeros((n * m, 48))
        V48 = np.zeros((m * p, 48))
        W48 = np.zeros((p * n, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i * m + j, col] = 1
                    V48[j * p + k, col] = 1
                    W48[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U48, V48, W48
        best_rank = 48

    # Try to reduce rank iteratively using the systematic elimination approach
    if time.time() < deadline - 30:
        U_try, V_try, W_try = _try_reduce_rank(
            best_U, best_V, best_W, n, m, p,
            max_reductions=best_rank - 30,
            time_budget=40
        )
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]

    # Try randomized reduction if there's still time
    if time.time() < deadline - 15:
        U_try, V_try, W_try = _try_randomized_reduction(
            best_U, best_V, best_W, n, m, p,
            time_budget=30
        )
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
