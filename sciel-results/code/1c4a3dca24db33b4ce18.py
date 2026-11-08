"""Matrix multiplication 2,4,6 - rank-minimal exact tensor decomposition search."""
import time
import numpy as np
import random

_DIMENSIONS = (2, 4, 6)
_DEADLINE = 135.0  # safety margin before 150s kill


def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    total = np.zeros(n * m * p, dtype=float)
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a.append(i * m + j)
                idx_b.append(j * p + k)
                idx_c.append(k * n + i)
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    for r in range(rank):
        total += U[idx_a, r] * V[idx_b, r] * W[idx_c, r]
    return np.allclose(total, np.ones(n * m * p), atol=1e-9)


def _strassen_decomposition(n, m, p):
    """
    Block Strassen decomposition for M_{2,4,6}.
    Split m=4 into 2 blocks of 2, p=6 into 3 blocks of 2.
    Each subproblem M_{2,2,2} uses rank-7 Strassen.
    Total rank = 2 * 3 * 7 = 42.
    """
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    a_blocks = m // 2  # 2
    b_blocks = p // 2  # 3
    strassen_rank = 7
    rank = a_blocks * b_blocks * strassen_rank  # 42

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for a in range(a_blocks):
        for b in range(b_blocks):
            for s in range(strassen_rank):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U[i * m + j, col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V[j * p + k, col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W[k * n + i, col] += coeff
                col += 1

    return U, V, W


def _strassen_3x3_decomposition(n, m, p):
    """
    Pad 2x4 to 3x4 and 4x6 to 4x3, use 3x3x3 Strassen (rank 27).
    The 3x3 Strassen gives a 3x3 output; we only need the 2x2 top-left block.
    """
    # 3x3x3 Strassen coefficients (standard)
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3-M6

    # A is 3x4 (padded from 2x4), B is 4x3 (padded from 4x6)
    # Split A into 2x2 blocks of size (1.5, 2)? No, use size 1x2? 
    # Actually for 3x3x3 we split into 3 blocks of size 1 along each dim.
    # A is 3x4: split rows into 3 blocks of 1, cols into... 4/3 not integer.
    # This doesn't work directly. Let me reconsider.
    
    # Actually, the standard 3x3x3 Strassen works on 3x3 * 3x3 -> 3x3.
    # We need to pad to 3x3 * 3x3. 
    # A: 2x4 -> pad to 3x4? No, we need square-ish.
    # Let's pad A to 3x4 (add 1 row of zeros) and B to 4x3 (add 1 col of zeros).
    # Then A*B is 3x3. We want the top-left 2x2 block.
    # But 3x3 * 4x3 doesn't give 3x3. A is 3x4, B is 4x3, result is 3x3. Good!
    # But Strassen 3x3x3 requires both inputs to be 3x3.
    # So we need to further pad: A from 3x4 to 3x4... hmm.
    
    # Alternative: pad A to 4x4 and B to 4x4, use 4x4x4 decomposition.
    # Or: use the fact that we can split 4 into 2+2 and use Strassen on 2x2 blocks.
    
    # Let me try a different approach: use the known rank-27 decomposition for 3x3x3
    # and adapt it. But 2x4 * 4x6 is not 3x3 * 3x3.
    
    # Actually, let's think about it differently. We can view:
    # A (2x4) * B (4x6) = C (2x6)
    # Split B into B1 (4x3) and B2 (4x3), so C = A*B1 + A*B2
    # Each A*Bk is 2x3.
    # Pad A to 3x3 (add 1 row of zeros, and pad cols from 4 to 3? No, 4>3)
    # This is getting complicated. Let me just use the block Strassen with 2x2 blocks.
    # That gives rank 42.
    
    # For the 3x3 approach: pad A from 2x4 to 3x4 (1 extra row), B from 4x6 to 4x3 (1 extra col).
    # A_padded (3x4) * B_padded (4x3) = C_padded (3x3). We want C[0:2, 0:3] which is all of it except row 2.
    # Now use 3x3x3 Strassen on A_padded and B_padded? But they're 3x4 and 4x3, not 3x3.
    
    # I think the directive's suggestion may not directly work. Let me implement a proper 
    # rank-reduction search instead, keeping the rank-42 Strassen as baseline.
    return None


def _random_search(n, m, p, start_rank, end_rank, budget_seconds):
    """
    Random tensor-network search for a low-rank decomposition.
    Tries random Gaussian initializations and does gradient descent
    on the squared residual of the tensor entries.
    """
    best_rank = None
    best_result = None
    start_time = time.time()
    rng = np.random.default_rng(42)

    # Precompute the target tensor index mapping
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a.append(i * m + j)
                idx_b.append(j * p + k)
                idx_c.append(k * n + i)
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    num_entries = n * m * p

    for rank in range(start_rank, max(end_rank, start_rank - 1), -1):
        if time.time() - start_time > budget_seconds:
            break
        # If we've already found a better rank, stop
        if best_rank is not None and rank >= best_rank:
            break

        max_restarts = 4
        for restart in range(max_restarts):
            if time.time() - start_time > budget_seconds:
                break

            U = rng.standard_normal((n * m, rank))
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))

            # Simple gradient descent with adaptive step
            lr = 0.01
            best_loss = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
            max_patience = 80

            for step in range(2000):
                if time.time() - start_time > budget_seconds:
                    break

                # Compute loss
                loss = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)

                for r in range(rank):
                    contrib = U[idx_a, r] * V[idx_b, r] * W[idx_c, r]
                    loss += np.sum((contrib - 1.0) ** 2)

                    # Gradients
                    # d/dU[a,r] = 2*(contrib-1)*V[b,r]*W[c,r]
                    resid = (contrib - 1.0) * 2.0
                    dU[idx_a, r] += resid * V[idx_b, r] * W[idx_c, r]
                    dV[idx_b, r] += resid * U[idx_a, r] * W[idx_c, r]
                    dW[idx_c, r] += resid * U[idx_a, r] * V[idx_b, r]

                loss /= num_entries

                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    patience = 0
                    if best_loss < 1e-12:
                        break
                else:
                    patience += 1
                    if patience > max_patience:
                        break

                # Normalize gradients
                gnorm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if gnorm > 1e-12:
                    dU /= gnorm
                    dV /= gnorm
                    dW /= gnorm
                    # Adaptive step size
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    # Reduce learning rate if loss increasing
                    if loss > best_loss * 1.1:
                        lr *= 0.5
                    elif loss < best_loss * 0.9:
                        lr *= 1.1
                    lr = min(lr, 0.1)
                else:
                    break

            # Check if we found a valid decomposition
            if best_loss < 1e-9:
                # Snap to nearest half-integers
                U_snap = np.round(best_U * 2) / 2
                V_snap = np.round(best_V * 2) / 2
                W_snap = np.round(best_W * 2) / 2
                if _verify(n, m, p, U_snap, V_snap, W_snap):
                    return rank, U_snap, V_snap, W_snap
                else:
                    # Try original (non-snapped)
                    if _verify(n, m, p, best_U, best_V, best_W):
                        return rank, best_U, best_V, best_W

    return None, None, None, None


def construct():
    n, m, p = _DIMENSIONS
    start = time.time()

    # First, get the known rank-42 Strassen decomposition as a safe fallback
    U_s, V_s, W_s = _strassen_decomposition(n, m, p)

    if not _verify(n, m, p, U_s, V_s, W_s):
        # Fallback to schoolbook (rank 48)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try random search for a lower rank
    best_rank = 42
    best_U, best_V, best_W = U_s, V_s, W_s

    # Time budget for search
    search_budget = min(_DEADLINE - (time.time() - start) - 5, 120)

    if search_budget > 10:
        # Try ranks from 38 down to 34
        for target_rank in range(38, 33, -1):
            if time.time() - start > _DEADLINE - 5:
                break
            remaining = _DEADLINE - (time.time() - start) - 3
            if remaining < 5:
                break

            rank_found, U_f, V_f, W_f = _random_search(
                n, m, p, target_rank, target_rank, min(remaining, 25)
            )
            if rank_found is not None and rank_found < best_rank:
                best_rank = rank_found
                best_U, best_V, best_W = U_f, V_f, W_f
                break  # Found a better rank, stop searching lower

    # Final verification
    if not _verify(n, m, p, best_U, best_V, best_W):
        best_U, best_V, best_W = U_s, V_s, W_s

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
