"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    deadline = time.time() + 140  # 20s safety margin

    # Try to find a rank-17 solution
    result = _find_rank17(n, m, p, deadline)
    if result is not None:
        return result

    # Fallback: schoolbook rank 18
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

def _find_rank17(n, m, p, deadline):
    """Search for a rank-17 half-integer decomposition."""
    rank = 17
    np.random.seed(42)

    for attempt in range(300):
        if time.time() > deadline:
            break

        # Random half-integer initialization for U and V
        scale = 3
        U = np.random.randint(-scale, scale + 1, size=(n * m, rank)) / 2.0
        V = np.random.randint(-scale, scale + 1, size=(m * p, rank)) / 2.0

        # Solve for W and check consistency
        W = np.zeros((p * n, rank))
        valid = True
        for i in range(n):
            for k in range(p):
                M = np.zeros((m * m, rank))
                for j1 in range(m):
                    for j2 in range(m):
                        M[j1 * m + j2, :] = U[i * m + j1, :] * V[j2 * p + k, :]

                rhs = np.zeros(m * m)
                for j in range(m):
                    rhs[j * m + j] = 1.0

                # Check consistency: rank(M) == rank([M|rhs])
                rank_M = np.linalg.matrix_rank(M)
                rank_aug = np.linalg.matrix_rank(np.hstack([M, rhs.reshape(-1, 1)]))
                if rank_M != rank_aug:
                    valid = False
                    break

                # Find minimum-norm solution
                W_ik = np.linalg.lstsq(M, rhs, rcond=None)[0]

                # Check if W_ik is close to half-integers
                W_ik_doubled = W_ik * 2
                if not np.allclose(W_ik_doubled, np.round(W_ik_doubled), atol=1e-6):
                    valid = False
                    break

                W[k * n + i, :] = np.round(W_ik_doubled) / 2.0
            if not valid:
                break

        if valid:
            # Final verification
            ok = True
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            s = np.sum(W[k*n+i, :] * U[i*m+j1, :] * V[j2*p+k, :])
                            expected = 1.0 if j1 == j2 else 0.0
                            if abs(s - expected) > 1e-9:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break

            if ok:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return None
# EVOLVE-BLOCK-END
