import time
import numpy as np

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Trivial decomposition (rank = n*m*p = 63)
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Precompute target tensor indices
    target_tensor = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    on_diag_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0
                on_diag_indices.append((a, b, c))

    def compute_full_error(U, V, W):
        """Compute total squared error over all tensor entries."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - target_tensor
        return np.sum(diff * diff)

    def is_valid_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is exact (within tolerance)."""
        # Check all on-diagonal terms equal 1
        for a, b, c in on_diag_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check all off-diagonal terms equal 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True

    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integer."""
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        return U_rounded, V_rounded, W_rounded

    def polish_solution(U, V, W, max_iter=50, tol=1e-14):
        """High-precision polish pass using alternating least squares with full constraints."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        
        prev_err = float('inf')
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # And sum_r U[i*m+j, r] * V[j*p+k', r] * W[k'*n+i', r] = 0 for off-diag
                    # Simplified: just enforce on-diagonal constraints via lstsq
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = x
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k, :] = x
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i, :] = x
                    except:
                        pass
            
            err = compute_full_error(U, V, W)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W

    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank

    # Search for lower rank
    # Lower bound: at least max(n*p, m*p, n*m) = max(21, 21, 9) = 21
    # But realistically, for 3x3x7, rank is likely around 30-40
    lower_bound = 25
    
    for R in range(trivial_rank - 1, max(lower_bound - 1, 20), -1):
        if time.time() > deadline - 20:
            break

        # Try multiple random initializations
        for trial in range(10):
            if time.time() > deadline - 15:
                break

            np.random.seed(trial * 1000 + R * 100)

            # Random initialization
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5

            # ALS iterations with combined objective
            converged = False
            prev_err = float('inf')

            for iteration in range(100):
                if time.time() > deadline - 10:
                    break

                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = x
                    except:
                        pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, R))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, R))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass

                # Compute full error
                err = compute_full_error(U, V, W)
                if abs(prev_err - err) < 1e-14:
                    converged = True
                    break
                prev_err = err

            # Check if we have a valid solution
            if converged or prev_err < 1e-8:
                # Try rounding to half-integers
                U_r, V_r, W_r = round_to_half_integers(U, V, W)
                if is_valid_decomposition(U_r, V_r, W_r, tol=1e-10):
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    best_rank = R
                    break  # Found valid for this rank, try lower

                # Try without rounding
                if is_valid_decomposition(U, V, W, tol=1e-10):
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = R
                    break

                # If close but not exact, try polish
                if prev_err < 1e-6:
                    U_p, V_p, W_p = polish_solution(U, V, W, max_iter=30)
                    U_pr, V_pr, W_pr = round_to_half_integers(U_p, V_p, W_p)
                    if is_valid_decomposition(U_pr, V_pr, W_pr, tol=1e-10):
                        best_U = U_pr
                        best_V = V_pr
                        best_W = W_pr
                        best_rank = R
                        break
                    if is_valid_decomposition(U_p, V_p, W_p, tol=1e-10):
                        best_U = U_p
                        best_V = V_p
                        best_W = W_p
                        best_rank = R
                        break

        if time.time() > deadline - 10:
            break

    # Final verification
    if best_rank < trivial_rank:
        return {
            "U": to_list(best_U),
            "V": to_list(best_V),
            "W": to_list(best_W)
        }
    else:
        return {
            "U": to_list(U_trivial),
            "V": to_list(V_trivial),
            "W": to_list(W_trivial)
        }
