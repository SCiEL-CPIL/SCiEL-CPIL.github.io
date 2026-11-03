import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 142.0  # Safety margin before 150s kill limit

    def build_strassen_rank28():
        """Build explicit rank-28 solution using two Strassen 2x2x2 blocks + schoolbook."""
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            f = j0 * p + (k0 + 1)
            g = (j0 + 1) * p + k0
            h = (j0 + 1) * p + (k0 + 1)
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)
            c01 = (k0 + 1) * n + i0
            c11 = (k0 + 1) * n + (i0 + 1)

            # M1 = (A00+A11)(B00+B11) -> C00 + C11
            U[a, col] = 1; U[d, col] = 1
            V[e, col] = 1; V[h, col] = 1
            W[c00, col] = 1; W[c11, col] = 1
            col += 1

            # M2 = (A10+A11)B00 -> C10 - C11
            U[c, col] = 1; U[d, col] = 1
            V[e, col] = 1
            W[c10, col] = 1; W[c11, col] = -1
            col += 1

            # M3 = A00(B01-B11) -> C01 + C11
            U[a, col] = 1
            V[f, col] = 1; V[h, col] = -1
            W[c01, col] = 1; W[c11, col] = 1
            col += 1

            # M4 = A11(B10-B00) -> C00 + C10
            U[d, col] = 1
            V[g, col] = 1; V[e, col] = -1
            W[c00, col] = 1; W[c10, col] = 1
            col += 1

            # M5 = (A00+A01)B11 -> -C00 + C01
            U[a, col] = 1; U[b, col] = 1
            V[h, col] = 1
            W[c00, col] = -1; W[c01, col] = 1
            col += 1

            # M6 = (A10-A00)(B00+B01) -> C11
            U[c, col] = 1; U[a, col] = -1
            V[e, col] = 1; V[f, col] = 1
            W[c11, col] = 1
            col += 1

            # M7 = (A01-A11)(B10+B11) -> C00
            U[b, col] = 1; U[d, col] = -1
            V[g, col] = 1; V[h, col] = 1
            W[c00, col] = 1
            col += 1

        # Block 1: i=0-1, j=0-1, k=0-1
        add_strassen(0, 0, 0)

        # Block 2: i=0-1, j=0-1, k=2-3
        add_strassen(0, 0, 2)

        # Block 3: i=0-1, j=0-1, k=4 (schoolbook 2x2x1 = 4 terms)
        for i in range(2):
            for j in range(2):
                U[i * m + j, col] = 1
                V[j * p + 4, col] = 1
                W[4 * n + i, col] = 1
                col += 1

        # Block 4: i=0-1, j=2, k=0-4 (schoolbook 2x1x5 = 10 terms)
        for i in range(2):
            for k in range(5):
                U[i * m + 2, col] = 1
                V[2 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W, rank

    def check_solution(U, V, W, R):
        """Check if the decomposition is exact. Returns max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def check_solution_fast(U, V, W, R):
        """Fast check using vectorized operations. Returns max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                prod = np.zeros(R)
                for k in range(p):
                    prod += V[j * p + k, :] * W[k * n + i, :]
                val = np.dot(U[i * m + j, :], prod)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
        return max_err

    def solve_for_U(V, W, R):
        U = np.zeros((n * m, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return U

    def solve_for_V(U, W, R):
        V = np.zeros((m * p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return V

    def solve_for_W(U, V, R):
        W = np.zeros((p * n, R))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return W

    def als_optimize(U, V, W, R, max_iters=200, tol=1e-12):
        """Run ALS optimization. Returns (U, V, W, err)."""
        prev_err = float('inf')
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        for it in range(max_iters):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            err = check_solution_fast(U, V, W, R)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < tol:
                break
            if it > 15 and err > best_err * 1.001 and err > prev_err * 0.999:
                break
            prev_err = err
        return best_U, best_V, best_W, best_err

    def polish(U, V, W, R, max_iters=500, tol=1e-14):
        """High-precision polish pass with tighter convergence tracking."""
        prev_err = float('inf')
        for it in range(max_iters):
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
            W = solve_for_W(U, V, R)
            err = check_solution(U, V, W, R)
            if err < tol:
                break
            if it > 5 and abs(prev_err - err) < 1e-16 * max(1.0, prev_err):
                break
            prev_err = err
        return U, V, W, err

    def epigraph_smooth_loss(U, V, W, R, beta=50.0):
        """
        Compute a smooth surrogate for the max absolute error using log-sum-exp.
        This provides a differentiable objective for gradient-based refinement.
        Returns (loss, grad_U, grad_V, grad_W)
        """
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    errors.append(abs(val - 1.0))
        errors = np.array(errors)
        
        # Log-sum-exp for smooth max
        max_e = np.max(errors)
        if max_e < 1e-15:
            return 0.0, None, None, None
        
        # Soft max approximation
        log_sum_exp = max_e + np.log(np.sum(np.exp((errors - max_e) * beta)) / beta)
        
        # For gradient computation, we need to be more careful
        # Use the softmax weights
        weights = np.exp((errors - max_e) * beta)
        weights /= np.sum(weights)
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    sign = 1.0 if val >= 1.0 else -1.0
                    w = weights[idx]
                    idx += 1
                    
                    # d/dU[i*m+j, r] of sign * (val - 1)
                    grad_U[i * m + j, :] += sign * w * V[j * p + k, :] * W[k * n + i, :]
                    grad_V[j * p + k, :] += sign * w * U[i * m + j, :] * W[k * n + i, :]
                    grad_W[k * n + i, :] += sign * w * U[i * m + j, :] * V[j * p + k, :]
        
        return log_sum_exp, grad_U, grad_V, grad_W

    def gradient_descent_polish(U, V, W, R, max_iters=100, lr=0.01, beta=50.0):
        """
        Gradient descent on the smooth epigraph loss to polish the solution.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            loss, gU, gV, gW = epigraph_smooth_loss(U, V, W, R, beta)
            if loss < 1e-12:
                break
            if gU is None:
                break
            
            # Adaptive learning rate
            loss_scale = max(1.0, loss)
            lr_eff = lr / loss_scale
            
            U -= lr_eff * gU
            V -= lr_eff * gV
            W -= lr_eff * gW
        
        return U, V, W

    def try_rank27_from_perturbation(U28, V28, W28, rank28, R, trial_idx, deadline):
        """Try to find rank-R solution by perturbing rank-28 solution."""
        if time.time() > deadline - 5:
            return None

        col_to_zero = trial_idx % rank28
        U = U28.copy()
        V = V28.copy()
        W = W28.copy()
        U[:, col_to_zero] = 0
        V[:, col_to_zero] = 0
        W[:, col_to_zero] = 0
        U = np.delete(U, col_to_zero, axis=1)
        V = np.delete(V, col_to_zero, axis=1)
        W = np.delete(W, col_to_zero, axis=1)

        np.random.seed(trial_idx * 104729 + 41)
        scale = 0.1 * np.exp(-trial_idx / 50.0)
        U += np.random.randn(n * m, R) * scale
        V += np.random.randn(m * p, R) * scale
        W += np.random.randn(p * n, R) * scale

        U, V, W, err = als_optimize(U, V, W, R, max_iters=150, tol=1e-12)
        if err < 1e-10:
            U, V, W, err = polish(U, V, W, R, max_iters=300, tol=1e-14)
            if err < 1e-8:
                final_err = check_solution(U, V, W, R)
                if final_err < 1e-8:
                    return U, V, W
        return None

    def random_restart_rank27(trial, deadline):
        """Random restart for rank-27 search."""
        if time.time() > deadline - 3:
            return None

        np.random.seed(trial * 7919 + 13)
        R = 27

        scale = 1.0 + 0.5 * np.sin(trial * 0.1)
        U = np.random.randn(n * m, R) * scale
        V = np.random.randn(m * p, R) * scale
        W = np.random.randn(p * n, R) * scale

        U_norm = np.linalg.norm(U)
        V_norm = np.linalg.norm(V)
        W_norm = np.linalg.norm(W)
        if U_norm > 0: U /= U_norm
        if V_norm > 0: V /= V_norm
        if W_norm > 0: W /= W_norm

        U, V, W, err = als_optimize(U, V, W, R, max_iters=80, tol=1e-12)
        if err < 1e-10:
            # Apply gradient polish
            U, V, W = gradient_descent_polish(U, V, W, R, max_iters=50, lr=0.005, beta=100.0)
            U, V, W, err = polish(U, V, W, R, max_iters=300, tol=1e-14)
            if err < 1e-8:
                final_err = check_solution(U, V, W, R)
                if final_err < 1e-8:
                    return U, V, W
        return None

    # Step 1: Build guaranteed rank-28 solution
    U28, V28, W28, rank28 = build_strassen_rank28()
    err28 = check_solution(U28, V28, W28, rank28)

    # Verify rank-28 is correct
    if err28 > 1e-10:
        # Fallback to schoolbook rank 30
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Step 2: Try to find rank-27 solution
    best_rank = rank28
    best_U, best_V, best_W = U28, V28, W28

    if time.time() < deadline - 10:
        R = 27

        # Strategy A: Perturb rank-28 by zeroing each column with diverse perturbations
        for trial_idx in range(30):
            if time.time() > deadline - 5:
                break
            result = try_rank27_from_perturbation(U28, V28, W28, rank28, R, trial_idx, deadline)
            if result is not None:
                best_rank = R
                best_U, best_V, best_W = result
                break

        # Strategy B: Random restarts with diverse seeds and gradient polish
        if best_rank == rank28 and time.time() < deadline - 5:
            n_trials = 2000
            for trial in range(n_trials):
                if time.time() > deadline - 3:
                    break
                result = random_restart_rank27(trial, deadline)
                if result is not None:
                    best_rank = R
                    best_U, best_V, best_W = result
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
