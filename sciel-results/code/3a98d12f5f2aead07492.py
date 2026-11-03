"""Matrix multiplication 2x3x6 - rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    def schoolbook():
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

    best_result = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def to_numpy(U, V, W):
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    def compute_error(U, V, W):
        """Compute the squared error of the decomposition."""
        U_np, V_np, W_np = to_numpy(U, V, W)
        r = U_np.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i * m + j], V_np[j * p + k])
                    val = np.dot(val, W_np[k * n + i])
                    err += (val - 1.0) ** 2
        return err

    def solve_2x3x3(rank, iterations=1000):
        """Search for rank-r decomposition of 2x3x3 multiplication."""
        nn, mm, pp = 2, 3, 3
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(iterations):
            if time.time() > deadline - 30:
                break
                
            U = np.random.randn(nn * mm, rank)
            V = np.random.randn(mm * pp, rank)
            W = np.random.randn(pp * nn, rank)
            
            # Local search refinement
            step_size = 0.1
            for _ in range(50):
                U_new = U + np.random.randn(*U.shape) * step_size
                V_new = V + np.random.randn(*V.shape) * step_size
                W_new = W + np.random.randn(*W.shape) * step_size
                
                err = compute_error(U_new.tolist(), V_new.tolist(), W_new.tolist())
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                    U, V, W = U_new, V_new, W_new
                    step_size *= 0.9
                else:
                    step_size = min(step_size * 1.1, 1.0)
                    
            if best_err < 1e-10:
                return best_U, best_V, best_W

        return None

    def combine_2x3x3_to_2x3x6(U1, V1, W1):
        """Combine two 2x3x3 rank-11 solutions into one 2x3x6 rank-22 solution."""
        U = np.hstack([U1, U1])
        V = np.zeros((18, 22))
        W = np.zeros((12, 22))
        
        for j in range(3):
            for k in range(3):
                V[j*6 + k, 0:11] = V1[j*3 + k, :]
                V[j*6 + k + 3, 11:22] = V1[j*3 + k, :]
                
        for k in range(3):
            for i in range(2):
                W[k*2 + i, 0:11] = W1[k*2 + i, :]
                W[(k+3)*2 + i, 11:22] = W1[k*2 + i, :]
                
        return U, V, W

    # Strategy 1: Try to find rank 11 for 2x3x3 and combine to rank 22 for 2x3x6
    if time.time() < deadline - 40:
        sol_233 = solve_2x3x3(11, iterations=500)
        if sol_233:
            U1, V1, W1 = sol_233
            U, V, W = combine_2x3x3_to_2x3x6(U1, V1, W1)
            if verify(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = 22
                return best_result

    # Strategy 2: Fallback to original random search for full problem
    def random_search(rank, iterations=10000):
        """Try to find a rank-r decomposition via random search."""
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None

        for it in range(iterations):
            if time.time() > deadline - 5:
                break

            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)

            err = compute_error(U.tolist(), V.tolist(), W.tolist())

            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()

            if err < 1e-12:
                return best_U, best_V, best_W, err

        return best_U, best_V, best_W, best_err

    def local_search(U_init, V_init, W_init, rank, iterations=5000, step_size=0.1):
        """Local search to refine a starting point."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        best_err = compute_error(U.tolist(), V.tolist(), W.tolist())

        for it in range(iterations):
            if time.time() > deadline - 5:
                break

            # Try random perturbations
            for _ in range(3):
                U_new = U + np.random.randn(*U.shape) * step_size
                V_new = V + np.random.randn(*V.shape) * step_size
                W_new = W + np.random.randn(*W.shape) * step_size

                err = compute_error(U_new.tolist(), V_new.tolist(), W_new.tolist())

                if err < best_err:
                    best_err = err
                    U = U_new
                    V = V_new
                    W = W_new
                    step_size *= 0.95
                else:
                    step_size *= 1.05
                    step_size = min(step_size, 1.0)

            if best_err < 1e-12:
                break

        return U, V, W, best_err

    # Try ranks from 20 up to 35
    for rank in range(20, min(36, best_rank)):
        if time.time() > deadline - 10:
            break

        # Random search
        U, V, W, err = random_search(rank, iterations=2000)

        if err is not None and err < 1e-9:
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break

        # Local search from best random point
        if U is not None:
            U, V, W, err = local_search(U, V, W, rank, iterations=1000)
            if err < 1e-9:
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break

    # If we found something better, return it
    if best_rank < n * m * p:
        return best_result

    # Otherwise return schoolbook
    return best_result
# EVOLVE-BLOCK-END
