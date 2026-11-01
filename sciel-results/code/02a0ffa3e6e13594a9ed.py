import time
import numpy as np

def construct():
    start = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = 2, 3, 5

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        R = U.shape[1]
        # Check valid entries: T[i*m+j, j*p+k, k*n+i] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check invalid entries: T[a, b, c] should equal 0 when indices don't align
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i, j = divmod(a, m)
                    jj, k = divmod(b, p)
                    kk, ii = divmod(c, n)
                    if not (i == ii and j == jj and k == kk):
                        val = 0.0
                        for r in range(R):
                            val += U[a, r] * V[b, r] * W[c, r]
                        if abs(val) > 1e-9:
                            return False
        return True

    # Trivial rank-30 decomposition as fallback
    R_trivial = n * m * p
    U_trivial = np.zeros((n * m, R_trivial), dtype=float)
    V_trivial = np.zeros((m * p, R_trivial), dtype=float)
    W_trivial = np.zeros((p * n, R_trivial), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, c] = 1.0
                V_trivial[j * p + k, c] = 1.0
                W_trivial[k * n + i, c] = 1.0
                c += 1

    # Try to find a lower-rank decomposition using random search
    # The tensor is sparse: only 30 non-zero entries in the 6x15x10 space
    # We want to find U (6,R), V (15,R), W (10,R) such that the decomposition is exact

    # Strategy: Use structured random initialization and local search
    # Try various ranks from low to high

    best_result = None
    best_rank = R_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial

    # First, try rank 20, 25, 28 with random search
    for target_rank in [20, 22, 25, 28]:
        if time.time() > deadline - 30:
            break

        # Try multiple random restarts
        max_restarts = 50 if target_rank <= 22 else 30
        found = False

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            R = target_rank
            # Initialize with random small values
            np.random.seed(42 + restart * 1000 + target_rank)
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5

            # Local search: minimize the error
            # Error = sum of squares of all T[a,b,c] values that should be 0,
            #         plus sum of squares of (T[i*m+j, j*p+k, k*n+i] - 1)
            def compute_error(U, V, W):
                err = 0.0
                R = U.shape[1]
                # Valid entries
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            err += (val - 1.0) ** 2
                # Invalid entries
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            i, j = divmod(a, m)
                            jj, k = divmod(b, p)
                            kk, ii = divmod(c, n)
                            if not (i == ii and j == jj and k == kk):
                                val = 0.0
                                for r in range(R):
                                    val += U[a, r] * V[b, r] * W[c, r]
                                err += val ** 2
                return err

            best_err = compute_error(U, V, W)

            # Iterative local search
            for it in range(200):
                if time.time() > deadline - 5:
                    break

                # Try perturbing each factor
                for factor in range(3):
                    if time.time() > deadline - 5:
                        break

                    if factor == 0:
                        idx = (it + restart) % (n * m)
                        r_idx = (it // (n * m)) % R
                        orig = U[idx, r_idx]
                        U[idx, r_idx] = orig + np.random.randn() * 0.1
                    elif factor == 1:
                        idx = (it + restart) % (m * p)
                        r_idx = (it // (m * p)) % R
                        orig = V[idx, r_idx]
                        V[idx, r_idx] = orig + np.random.randn() * 0.1
                    else:
                        idx = (it + restart) % (p * n)
                        r_idx = (it // (p * n)) % R
                        orig = W[idx, r_idx]
                        W[idx, r_idx] = orig + np.random.randn() * 0.1

                    new_err = compute_error(U, V, W)
                    if new_err < best_err:
                        best_err = new_err
                        if best_err < 1e-8:
                            break
                    else:
                        # Revert
                        if factor == 0:
                            U[idx, r_idx] = orig
                        elif factor == 1:
                            V[idx, r_idx] = orig
                        else:
                            W[idx, r_idx] = orig

                if best_err < 1e-8:
                    break

            if best_err < 1e-6:
                # Check if it's a valid exact decomposition
                if verify(U, V, W):
                    # Round to half-integers if possible
                    U_out = np.round(U * 2) / 2
                    V_out = np.round(V * 2) / 2
                    W_out = np.round(W * 2) / 2
                    if verify(U_out, V_out, W_out):
                        best_U, best_V, best_W = U_out, V_out, W_out
                    else:
                        best_U, best_V, best_W = U, V, W
                    best_rank = R
                    found = True
                    break

        if found:
            break

    # If we found a better decomposition, return it
    if best_rank < R_trivial:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Otherwise, return the trivial rank-30 decomposition
    return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
