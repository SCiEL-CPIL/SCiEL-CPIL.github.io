import numpy as np
import time

DIMENSIONS = [4, 4, 5]


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy einsum for speed with early termination."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    diff = np.abs(reconstructed - 1.0)
    if np.any(diff > tol):
        return False
    return np.allclose(reconstructed, 1.0, atol=tol)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 x 4x4 using 2x2 Strassen blocks."""
    block_coeffs_U = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    block_coeffs_V = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    block_coeffs_W = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(1, 0, 1), (0, 0, 1)],
        [(0, 1, -1), (1, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    rank = 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc

                for (br, bc, coeff_u) in block_coeffs_U[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            i = br * 2 + lr2
                            j = bc * 2 + lc2
                            U[i * 4 + j][c] += coeff_u

                for (br, bc, coeff_v) in block_coeffs_V[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            j = br * 2 + lr2
                            k = bc * 2 + lc2
                            V[j * 4 + k][c] += coeff_v

                for (br, bc, coeff_w) in block_coeffs_W[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            k = br * 2 + lr2
                            i = bc * 2 + lc2
                            W[k * 4 + i][c] += coeff_w

    return U, V, W


def _build_4x4x5_from_strassen():
    """Build 4x4x5 decomposition: Strassen 4x4x4 core + boundary terms."""
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28

    rank_boundary = 16
    rank_total = rank_core + rank_boundary

    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(20)]
    W = [[0] * rank_total for _ in range(20)]

    for c in range(rank_core):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][c] = U44[i * 4 + j][c]
        for j in range(4):
            for k in range(4):
                V[j * 5 + k][c] = V44[j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][c] = W44[k * 4 + i][c]

    # Add boundary terms for the 5th column
    for i in range(4):
        for j in range(4):
            c = rank_core + i * 4 + j
            U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            W[4 * 4 + i][c] = 1

    return U, V, W


def _term_sparsity(U, V, W, c):
    """Calculate sparsity score for term c - lower is more redundant."""
    U_nz = sum(1 for x in U[i][c] for i in range(len(U)) if abs(x) > 1e-9)
    V_nz = sum(1 for x in V[i][c] for i in range(len(V)) if abs(x) > 1e-9)
    W_nz = sum(1 for x in W[i][c] for i in range(len(W)) if abs(x) > 1e-9)
    return U_nz + V_nz + W_nz


def _try_reduce_rank_aggressive(U, V, W, n, m, p, max_iter=500, tol=1e-9, deadline=None):
    """Aggressive rank reduction with smarter ordering."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])

    if deadline is None:
        deadline = time.time() + 100

    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break

        removed = False

        # Calculate sparsity scores and sort terms (most redundant first)
        sparsity_scores = [_term_sparsity(U, V, W, c) for c in range(rank)]
        sorted_terms = sorted(range(rank), key=lambda c: sparsity_scores[c])

        # Phase 1: Try removing individual terms (most redundant first)
        for c in sorted_terms:
            if time.time() > deadline - 5:
                break
            U_test = [row[:c] + row[c + 1:] for row in U]
            V_test = [row[:c] + row[c + 1:] for row in V]
            W_test = [row[:c] + row[c + 1:] for row in W]
            if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                U = U_test
                V = V_test
                W = W_test
                rank -= 1
                removed = True
                break

        if removed:
            continue

        # Phase 2: Try removing pairs (most redundant pairs first)
        if rank > 2 and iteration % 3 == 0:
            top_terms = sorted_terms[:min(5, rank)]
            for i, c1 in enumerate(top_terms):
                if time.time() > deadline - 5:
                    break
                for c2 in top_terms[i + 1:]:
                    if time.time() > deadline - 5:
                        break
                    idxs = [j for j in range(rank) if j != c1 and j != c2]
                    U_test = [[U[i][j] for j in idxs] for i in range(n * m)]
                    V_test = [[V[i][j] for j in idxs] for i in range(m * p)]
                    W_test = [[W[i][j] for j in idxs] for i in range(p * n)]
                    if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                        U = U_test
                        V = V_test
                        W = W_test
                        rank -= 2
                        removed = True
                        break
                if removed:
                    break

        if not removed:
            break

    return U, V, W, rank


def _try_linear_combination_removal(U, V, W, n, m, p, tol=1e-9, deadline=None):
    """Try to find linear dependencies among rank-1 terms that allow reduction."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])

    if deadline is None:
        deadline = time.time() + 50

    for iteration in range(rank):
        if time.time() > deadline - 5:
            break

        removed = False
        for c1 in range(rank - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            for c2 in range(rank - 1, c1, -1):
                alpha = None
                for idx in range(n * m):
                    if abs(U[idx][c2]) > tol:
                        alpha = U[idx][c1] / U[idx][c2]
                        break

                if alpha is None:
                    continue

                match = True
                for idx in range(n * m):
                    if abs(U[idx][c1] - alpha * U[idx][c2]) > tol * 10:
                        match = False
                        break
                if not match:
                    continue

                for idx in range(m * p):
                    if abs(V[idx][c1] - alpha * V[idx][c2]) > tol * 10:
                        match = False
                        break
                if not match:
                    continue

                for idx in range(p * n):
                    if abs(W[idx][c1] - alpha * W[idx][c2]) > tol * 10:
                        match = False
                        break
                if not match:
                    continue

                U_test = [row[:c1] + row[c1 + 1:] for row in U]
                V_test = [row[:c1] + row[c1 + 1:] for row in V]
                W_test = [row[:c1] + row[c1 + 1:] for row in W]
                if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                    U = U_test
                    V = V_test
                    W = W_test
                    rank -= 1
                    removed = True
                    break
            if removed:
                break

        if not removed:
            break

    return U, V, W, rank


def _try_column_merging(U, V, W, n, m, p, tol=1e-9, deadline=None):
    """Try to merge columns by finding linear combinations that preserve the tensor.
    
    This is a more sophisticated approach: for each pair of terms (c1, c2),
    check if there exists a linear combination alpha*c1 + beta*c2 that can
    replace both while preserving the decomposition.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])

    if deadline is None:
        deadline = time.time() + 50

    for iteration in range(rank):
        if time.time() > deadline - 5:
            break

        removed = False
        for c1 in range(rank - 1):
            if time.time() > deadline - 5:
                break
            for c2 in range(c1 + 1, rank):
                if time.time() > deadline - 5:
                    break

                # Try to find alpha, beta such that:
                # For all i,j,k: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
                # remains unchanged when we replace terms c1,c2 with a single term.
                # This requires:
                # U[i,j,c1]*V[j,k,c1]*W[k,i,c1] + U[i,j,c2]*V[j,k,c2]*W[k,i,c2]
                # = U_new[i,j]*V_new[j,k]*W_new[k,i]
                
                # Simple case: if one term is a scalar multiple of the other
                # in all three matrices, we can just keep one.
                # More generally, try small integer combinations.
                
                for alpha in [-2, -1, 0, 1, 2]:
                    for beta in [-2, -1, 0, 1, 2]:
                        if alpha == 0 and beta == 0:
                            continue
                        if alpha == 0:
                            # Just keep c2
                            continue
                        if beta == 0:
                            # Just keep c1
                            continue
                        
                        # Create new term
                        U_new = [U[i][c1] * alpha + U[i][c2] * beta for i in range(n * m)]
                        V_new = [V[i][c1] * alpha + V[i][c2] * beta for i in range(m * p)]
                        W_new = [W[i][c1] * alpha + W[i][c2] * beta for i in range(p * n)]
                        
                        # Check if this single term can replace both c1 and c2
                        # Build test matrices
                        idxs = [j for j in range(rank) if j != c1 and j != c2]
                        U_test = []
                        for i in range(n * m):
                            row = [U[i][j] for j in idxs]
                            row.append(U_new[i])
                            U_test.append(row)
                        V_test = []
                        for i in range(m * p):
                            row = [V[i][j] for j in idxs]
                            row.append(V_new[i])
                            V_test.append(row)
                        W_test = []
                        for i in range(p * n):
                            row = [W[i][j] for j in idxs]
                            row.append(W_new[i])
                            W_test.append(row)
                        
                        if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                            U = U_test
                            V = V_test
                            W = W_test
                            rank -= 1
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    break
            if removed:
                break

        if not removed:
            break

    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    # Build structured Strassen-based 4x4x5 decomposition
    if time.time() < deadline:
        U_new, V_new, W_new = _build_4x4x5_from_strassen()
        if _verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
            rank_total = len(U_new[0])
            best_rank = rank_total
            best_result = {"U": U_new, "V": V_new, "W": W_new}

            # Aggressive rank reduction
            if time.time() < deadline - 15:
                U_r, V_r, W_r, rank_r = _try_reduce_rank_aggressive(
                    U_new, V_new, W_new, n, m, p,
                    max_iter=1000, deadline=deadline - 10
                )
                if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                    best_rank = rank_r
                    best_result = {"U": U_r, "V": V_r, "W": W_r}

            # Try linear combination removal
            if time.time() < deadline - 5:
                U_r, V_r, W_r, rank_r = _try_linear_combination_removal(
                    best_result["U"], best_result["V"], best_result["W"],
                    n, m, p, deadline=deadline - 2
                )
                if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                    best_rank = rank_r
                    best_result = {"U": U_r, "V": V_r, "W": W_r}

            # Try column merging
            if time.time() < deadline - 5:
                U_r, V_r, W_r, rank_r = _try_column_merging(
                    best_result["U"], best_result["V"], best_result["W"],
                    n, m, p, deadline=deadline - 2
                )
                if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                    best_rank = rank_r
                    best_result = {"U": U_r, "V": V_r, "W": W_r}

    # Final reduction pass
    if time.time() < deadline - 5:
        U_best, V_best, W_best = best_result["U"], best_result["V"], best_result["W"]
        U_r, V_r, W_r, rank_r = _try_reduce_rank_aggressive(
            U_best, V_best, W_best, n, m, p,
            max_iter=300, deadline=deadline - 2
        )
        if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
            best_rank = rank_r
            best_result = {"U": U_r, "V": V_r, "W": W_r}

    return best_result
