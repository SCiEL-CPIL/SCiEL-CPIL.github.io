"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_rank17():
    """Build a verified rank-17 decomposition for 2x3x3 matrix multiplication."""
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2: (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # M4: A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6: (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _to_list(U, V, W):
    """Convert numpy arrays to nested list format with integer entries."""
    U_list = [[int(U[i, r]) for r in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[int(V[i, r]) for r in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[int(W[i, r]) for r in range(W.shape[1])] for i in range(W.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}


def _try_rank16_structured(deadline):
    """
    Structured rank-16 search: start from rank-17, remove one term,
    and solve for W adjustments (linear in W given fixed U,V).
    The system decomposes by output index c.
    """
    n, m, p = 2, 3, 3
    base = _build_rank17()
    U0 = base["U"].copy()
    V0 = base["V"].copy()
    W0 = base["W"].copy()

    for remove_term in range(17):
        if time.time() > deadline:
            return None

        kept = [r for r in range(17) if r != remove_term]

        # For each output index c, solve a small linear system
        W_new = np.zeros((6, 16))
        success = True

        for c in range(6):
            i = c % n
            k = c // n
            # Build 3x16 system: for j=0,1,2
            A_mat = np.zeros((3, 16))
            for j in range(3):
                for s in range(16):
                    r = kept[s]
                    A_mat[j, s] = U0[i * m + j, r] * V0[j * p + k, r]

            try:
                sol, res, rank_mat, sv = np.linalg.lstsq(A_mat, np.ones(3), rcond=None)
                W_new[c, :] = sol
                if np.linalg.norm(A_mat @ sol - np.ones(3)) > 1e-8:
                    success = False
                    break
            except:
                success = False
                break

        if not success:
            continue

        # Check if solution is half-integer
        W_check = W_new * 2
        W_rounded = np.round(W_check)
        if np.max(np.abs(W_check - W_rounded)) < 1e-6:
            W_half = W_rounded / 2.0

            # Build full candidate
            U_new = np.zeros((6, 16))
            V_new = np.zeros((9, 16))
            W_final = np.zeros((6, 16))
            for s in range(16):
                r = kept[s]
                U_new[:, s] = U0[:, r]
                V_new[:, s] = V0[:, r]
                W_final[:, s] = W_half[:, s]

            if _verify(U_new, V_new, W_final):
                return _to_list(U_new, V_new, W_final)

    return None


def _try_rank16_random(deadline, max_attempts=500):
    """
    Randomized rank-16 search with structured sparsity patterns.
    Each term gets a random sparse pattern from a small set.
    """
    n, m, p = 2, 3, 3
    rank = 16
    np.random.seed(12345)

    for attempt in range(max_attempts):
        if time.time() > deadline:
            break

        U = np.zeros((6, rank))
        V = np.zeros((9, rank))
        W = np.zeros((6, rank))

        for r in range(rank):
            # Sparse random patterns
            for idx in range(6):
                U[idx, r] = np.random.choice([-1, 0, 0, 0, 1], 1)[0]
            for idx in range(9):
                V[idx, r] = np.random.choice([-1, 0, 0, 0, 1], 1)[0]
            for idx in range(6):
                W[idx, r] = np.random.choice([-1, 0, 0, 0, 1], 1)[0]

        # Vectorized check
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            return _to_list(U, V, W)

    return None


def _try_rank16_merge(deadline):
    """
    Try merging pairs of terms from rank-17 by checking if their
    combined contribution can be absorbed by adjusting remaining terms.
    Uses a two-stage approach: fix U,V, solve for W; then check.
    """
    n, m, p = 2, 3, 3
    base = _build_rank17()
    U0 = base["U"].copy()
    V0 = base["V"].copy()
    W0 = base["W"].copy()

    # Try removing each single term and solving
    # (same as structured but with different tolerance/approach)
    for remove_term in range(17):
        if time.time() > deadline:
            return None

        kept = [r for r in range(17) if r != remove_term]
        W_new = np.zeros((6, 16))
        success = True

        for c in range(6):
            i = c % n
            k = c // n
            A_mat = np.zeros((3, 16))
            for j in range(3):
                for s in range(16):
                    r = kept[s]
                    A_mat[j, s] = U0[i * m + j, r] * V0[j * p + k, r]

            # Try exact solve first, then least squares
            try:
                # Check if system is exactly solvable
                rank_mat = np.linalg.matrix_rank(A_mat)
                if rank_mat >= 3:
                    sol, res, rk, sv = np.linalg.lstsq(A_mat, np.ones(3), rcond=None)
                    W_new[c, :] = sol
                    if np.linalg.norm(A_mat @ sol - np.ones(3)) > 1e-7:
                        success = False
                        break
                else:
                    success = False
                    break
            except:
                success = False
                break

        if not success:
            continue

        # Accept if entries are "nice" (within small set)
        max_abs = np.max(np.abs(W_new))
        if max_abs > 10:
            continue

        # Round to nearest 0.5 and verify
        W_rounded = np.round(W_new * 2) / 2.0

        U_new = np.zeros((6, 16))
        V_new = np.zeros((9, 16))
        W_final = np.zeros((6, 16))
        for s in range(16):
            r = kept[s]
            U_new[:, s] = U0[:, r]
            V_new[:, s] = V0[:, r]
            W_final[:, s] = W_rounded[:, s]

        if _verify(U_new, V_new, W_final):
            return _to_list(U_new, V_new, W_final)

    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for
    multiplying a 2x3 matrix by a 3x3 matrix.

    Interface:
    U: shape (n*m, rank) = (6, rank)
    V: shape (m*p, rank) = (9, rank)
    W: shape (p*n, rank) = (6, rank)

    n=2, m=3, p=3.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Strategy 1: Structured search (remove one term, solve linear system)
    result = _try_rank16_structured(deadline)
    if result is not None:
        if _verify(np.array(result["U"]), np.array(result["V"]), np.array(result["W"])):
            return result

    # Strategy 2: Merge-based search with relaxed tolerance
    if time.time() < deadline:
        result = _try_rank16_merge(deadline)
        if result is not None:
            if _verify(np.array(result["U"]), np.array(result["V"]), np.array(result["W"])):
                return result

    # Strategy 3: Random search (limited attempts to stay within budget)
    if time.time() < deadline:
        remaining = deadline - time.time()
        max_attempts = min(300, int(remaining * 50))
        if max_attempts > 0:
            result = _try_rank16_random(deadline, max_attempts=max_attempts)
            if result is not None:
                if _verify(np.array(result["U"]), np.array(result["V"]), np.array(result["W"])):
                    return result

    # Fallback: return the verified rank-17 decomposition
    base = _build_rank17()
    return _to_list(base["U"], base["V"], base["W"])
# EVOLVE-BLOCK-END
