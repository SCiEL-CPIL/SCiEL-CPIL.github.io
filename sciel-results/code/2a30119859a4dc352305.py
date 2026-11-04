import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s limit, 20s buffer

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, r):
        """Verify exact tensor decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+ip][c]
                        expected = 1.0 if i == ip else 0.0
                        if abs(s - expected) > 1e-9:
                            return False
        return True

    def to_clean_list(mat):
        """Convert to list of lists with int/half-int entries."""
        result = []
        for row in mat:
            clean_row = []
            for val in row:
                v2 = round(val * 2)
                if v2 % 2 == 0:
                    clean_row.append(v2 // 2)
                else:
                    clean_row.append(v2 / 2.0)
            result.append(clean_row)
        return result

    best_rank = n * m * p
    best_result = get_schoolbook()

    # Structural approach: exploit n=2 symmetry.
    # The tensor M_{2,4,7} requires: sum_c U[i,j,c]*V[j,k,c]*W[k,i',c] = delta(i,i')
    # For i != i', this is 0; for i == i', it's 1.
    # Key insight: W[k,0,:] and W[k,1,:] must be "orthogonal" to each other's
    # contractions. We can try to make W[k,0,:] and W[k,1,:] lie in orthogonal
    # subspaces by construction.
    
    # Try ranks from 15 to 56
    for r in range(15, min(57, n * m * p + 1)):
        if time.time() > deadline:
            break

        max_restarts = 120 if r <= 25 else 60

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Random U, V with entries in {-1, 0, 1}
            U_flat = np.random.randint(-1, 2, size=(n * m, r))
            V_flat = np.random.randint(-1, 2, size=(m * p, r))

            # Solve for W row by row
            W_flat = np.zeros((p * n, r))
            valid_w = True

            for k in range(p):
                for ip in range(n):
                    # Build linear system: A @ w = b
                    # A[i*m+j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # b[i*m+j] = 1 if i==ip else 0
                    A = np.zeros((n * m, r))
                    b = np.zeros(n * m)

                    for i in range(n):
                        for j in range(m):
                            A[i * m + j, :] = U_flat[i * m + j, :] * V_flat[j * p + k, :]
                            b[i * m + j] = 1.0 if i == ip else 0.0

                    # Check if system is consistent
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check residual
                        if rank_A < n * m and np.linalg.norm(A @ w - b) > 1e-8:
                            valid_w = False
                            break
                        W_flat[k * n + ip, :] = w
                    except:
                        valid_w = False
                        break
                if not valid_w:
                    break

            if not valid_w:
                continue

            # Round W to nearest half-integer
            W_rounded = np.round(W_flat * 2) / 2.0

            # Check if rounding is close
            if not np.allclose(W_flat, W_rounded, atol=1e-6):
                continue

            # Verify decomposition exactly
            if verify(U_flat.tolist(), V_flat.tolist(), W_rounded.tolist(), r):
                best_rank = r
                best_result = {
                    "U": to_clean_list(U_flat),
                    "V": to_clean_list(V_flat),
                    "W": to_clean_list(W_rounded)
                }
                break  # Found valid for this r, move to next r

        if best_rank == r:
            break  # Found a valid decomposition at rank r

    return best_result
# EVOLVE-BLOCK-END
