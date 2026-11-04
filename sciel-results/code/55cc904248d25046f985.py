import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    max_time = 140  # Safety margin before 150s kill

    n, m, p = 2, 3, 4
    # Target tensor: T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Shapes: U (n*m, R) = (6, R), V (m*p, R) = (12, R), W (p*n, R) = (8, R)
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]

    def verify_decomp(U, V, W, R):
        """Verify if decomposition is exact."""
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        T_target = np.ones((n, m, p))
        return np.allclose(A_approx, T_target, atol=1e-9)

    def snap_to_halfint(arr, tol=1e-6):
        """Snap array to nearest half-integer (0, 0.5, 1, 1.5, 2, ...)."""
        # Multiply by 2, round, divide by 2
        snapped = np.round(arr * 2) / 2.0
        return snapped

    def try_find_decomp(R, max_restarts=200, verbose=False):
        """Try to find rank R decomposition for 2x3x4."""
        num_params = R * (n * m + m * p + p * n)
        
        def unpack(params):
            U = params[:R * n * m].reshape(R, n * m).T
            V = params[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
            W = params[R * (n * m + m * p) :].reshape(R, p * n).T
            return U, V, W

        def residuals(params):
            U, V, W = unpack(params)
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return (np.ones((n, m, p)) - A_approx).flatten()

        from scipy.optimize import least_squares

        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                return None

            # Various initialization strategies
            if attempt % 5 == 0:
                init_params = np.random.uniform(-1, 1, size=num_params)
            elif attempt % 5 == 1:
                init_params = np.random.uniform(-2, 2, size=num_params)
            elif attempt % 5 == 2:
                # Sparse-like init
                init_params = np.zeros(num_params)
                for _ in range(num_params // 2):
                    idx = np.random.randint(0, num_params)
                    init_params[idx] = np.random.choice([-1, 1])
            elif attempt % 5 == 3:
                # Small perturbations around 0
                init_params = np.random.normal(0, 0.5, size=num_params)
            else:
                init_params = np.random.uniform(-0.5, 0.5, size=num_params)

            try:
                sol = least_squares(residuals, init_params, method='trf', max_nfev=100)
                
                if sol.cost < 1e-8:
                    # Snap to half-integers
                    rounded_params = snap_to_halfint(sol.x)
                    res_check = residuals(rounded_params)
                    if np.linalg.norm(res_check) < 1e-9:
                        U, V, W = unpack(rounded_params)
                        if verify_decomp(U, V, W, R):
                            return {"U": U, "V": V, "W": W}
            except Exception:
                pass

        return None

    # Try ranks from 19 upward (known lower bound for 2x3x4 is 19)
    # Actually, let's start from 18 to be safe, but 19 is the known lower bound
    for R in range(18, 25):
        if time.time() - start_time > max_time - 10:
            break
            
        if verbose:
            print(f"Trying rank {R}...")
        
        dec = try_find_decomp(R, max_restarts=150)
        
        if dec is not None:
            if verbose:
                print(f"Found rank {R} decomposition!")
            return {
                "U": dec["U"].tolist(),
                "V": dec["V"].tolist(),
                "W": dec["W"].tolist()
            }

    # Fallback to schoolbook rank 24
    if verbose:
        print("Falling back to schoolbook rank 24")
    rank = 24
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
