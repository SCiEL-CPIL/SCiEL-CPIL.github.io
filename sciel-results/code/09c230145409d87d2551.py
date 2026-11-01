<<<<<<< SEARCH
def _verify_np(U, V, W, tol=1e-9):
    """Verify decomposition using numpy. Returns True if exact within tol."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for r in range(rank):
                    s += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True
=======
def _verify_np(U, V, W, tol=1e-9):
    """Verify decomposition using numpy. Returns True if exact within tol."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for r in range(rank):
                    s += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _check_half_integer(U, V, W, tol=1e-9):
    """Check if all entries are integer or half-integer within tolerance."""
    for matrix in [U, V, W]:
        for val in matrix.flatten():
            # Check if val is close to k/2 for some integer k
            two_val = val * 2
            nearest = round(two_val)
            if abs(two_val - nearest) > tol:
                return False
    return True
>>>>>>> REPLACE

