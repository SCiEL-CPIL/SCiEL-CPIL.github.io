"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Build Strassen-based decomposition
    result = _build_strassen_3x5x6(n, m, p)

    if result is not None:
        U, V, W, rank = result
        if _verify_fast(U, V, W, n, m, p):
            # Aggressive rank reduction via linear dependency search
            U, V, W, rank = _reduce_rank_aggressive(U, V, W, rank, n, m, p, deadline)
            if _verify_fast(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                # Reduction failed, rebuild and do lighter reduction
                U, V, W, rank = _build_strassen_3x5x6(n, m, p)
                U, V, W, rank = _try_reduce_rank(U, V, W, rank, n, m, p, deadline)
                if _verify_fast(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _reduce_rank_aggressive(U, V, W, rank, n, m, p, deadline):
    """
    Iteratively search for linear dependencies among rank-1 tensor terms.
    If T_r = sum_{c != r} alpha_c * T_c, we can absorb T_r into the others.
    Repeat until no more dependencies are found or deadline is reached.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()

    def tensor_to_vec(u, v, w):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i * m + j) * p + k] = u[i] * v[j] * w[k]
        return vec

    # Build tensor vectors for all current columns
    def build_tensors(U, V, W):
        tensors = []
        for c in range(U.shape[1]):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            tensors.append(tensor_to_vec(u, v, w))
        return np.array(tensors)

    tensors = build_tensors(U, V, W)
    current_rank = U.shape[1]
    removed = set()

    # Iteratively remove dependent terms
    max_iterations = 20
    for iteration in range(max_iterations):
        if time.time() > deadline - 15:
            break

        found_removal = False
        for r in range(current_rank):
            if r in removed:
                continue
            if time.time() > deadline - 15:
                break

            others = [c for c in range(current_rank) if c != r and c not in removed]
            if len(others) < 1:
                continue

            A = tensors[others]
            b = tensors[r]

            try:
                # Solve A.T @ alpha = b  (least squares)
                ATA = A @ A.T
                Ab = A @ b

                # Check if system is consistent
                rank_ATA = np.linalg.matrix_rank(ATA, tol=1e-10)
                rank_AB = np.linalg.matrix_rank(np.hstack([ATA, Ab.reshape(-1, 1)]), tol=1e-10)

                if rank_ATA == rank_AB:
                    # Consistent: T_r is in span of others
                    alpha, _, _, _ = np.linalg.lstsq(ATA, Ab, rcond=1e-10)
                    residual = b - A.T @ alpha
                    if np.linalg.norm(residual) < 1e-8:
                        removed.add(r)
                        for idx, c in enumerate(others):
                            scale = 1.0 + alpha[idx]
                            U[:, c] *= scale
                        found_removal = True
                        # Rebuild tensors after modification
                        tensors = build_tensors(U, V, W)
                        current_rank = U.shape[1]
                        break
            except np.linalg.LinAlgError:
                continue

        if not found_removal:
            break

    if removed:
        keep = [c for c in range(U.shape[1]) if c not in removed]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        current_rank = len(keep)

    return U, V, W, current_rank


def _try_reduce_rank(U, V, W, rank, n, m, p, deadline):
    """
    Try to reduce rank by checking if any rank-1 tensor term lies in the span
    of the others.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()

    current_rank = rank
    removed = set()

    def tensor_to_vec(u, v, w):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i * m + j) * p + k] = u[i] * v[j] * w[k]
        return vec

    tensors = []
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        tensors.append(tensor_to_vec(u, v, w))

    tensors = np.array(tensors)

    for r in range(rank):
        if r in removed:
            continue
        if time.time() > deadline - 10:
            break

        others = [c for c in range(rank) if c != r and c not in removed]
        if len(others) < 1:
            continue

        A = tensors[others]
        b = tensors[r]

        try:
            ATA = A @ A.T
            Ab = A @ b

            alpha, residuals, rank_lsq, sv = np.linalg.lstsq(ATA, Ab, rcond=1e-10)

            residual = b - A.T @ alpha
            if np.linalg.norm(residual) < 1e-8:
                removed.add(r)
                for idx, c in enumerate(others):
                    scale = 1.0 + alpha[idx]
                    U[:, c] *= scale
                break
        except np.linalg.LinAlgError:
            continue

    if removed:
        keep = [c for c in range(rank) if c not in removed]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        current_rank = len(keep)

    return U, V, W, current_rank


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1

    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[2, 3] = 1; V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1

    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1

    return U, V, W


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen block structure.
    """
    sU, sV, sW = _get_strassen_2x2x2()

    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1

        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]
            U[0*5 + 1, col] = sU[1, c]
            U[1*5 + 0, col] = sU[2, c]
            U[1*5 + 1, col] = sU[3, c]
            V[0*6 + k0, col] = sV[0, c]
            V[0*6 + k1, col] = sV[1, c]
            V[1*6 + k0, col] = sV[2, c]
            V[1*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1

        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1

        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1

        # Bottom row (i=2), j in {0,1}: schoolbook 1x2x2
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1

        # Bottom row (i=2), j in {2,3}: schoolbook 1x2x2
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1

        # Bottom row (i=2), j=4: schoolbook 1x1x2
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1

    if col != rank:
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]

    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    r = U.shape[1]

    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)

    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)

    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
