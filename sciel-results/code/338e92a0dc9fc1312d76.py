"""Combined program: structural block decomposition + randomized ALS search with deadline guard."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 115  # safety margin for 120s evaluator

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_time_limit = START_TIME + TIME_LIMIT

    def time_remaining():
        return total_time_limit - time.time()

    def check_time():
        return time_remaining() < 5.0

    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m + j, j*p + k, k*n + i] = 1 for valid i,j,k
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Fallback: schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Check if U, V, W give exact decomposition."""
        r = len(U[0]) if U else 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = 0.0
                    for ri in range(r):
                        val += U[a][ri] * V[b][ri] * W[c][ri]
                    if abs(val - T[a, b, c]) > 1e-9:
                        return False
        return True

    def verify_fast(U, V, W):
        """Vectorized verification."""
        r = len(U[0]) if U else 0
        # Compute T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum or manual
        T_rec = np.zeros((nm, mp, pn), dtype=float)
        for ri in range(r):
            T_rec += U[:, ri].reshape(nm, 1, 1) * V[:, ri].reshape(1, mp, 1) * W[:, ri].reshape(1, 1, pn)
        return np.allclose(T_rec, T, atol=1e-9)

    # Try block-based construction: split 6 columns into two 3-column blocks
    # For each block, use a 3x3x3 decomposition
    # This gives a structured starting point

    # Try randomized ALS for rank r, from r=38 down to r=53
    # Start from a reasonable upper bound
    best_result = None
    best_rank = n * m * p  # 54

    # Try ranks from 40 to 53 (skip some for time)
    candidate_ranks = list(range(40, best_rank))  # 40..53
    # Add some lower ranks if time permits
    candidate_ranks += [38, 36, 34, 32, 30, 28, 26]

    for r in candidate_ranks:
        if check_time():
            break
        if r >= best_rank:
            continue

        # Try a few random initializations
        n_attempts = 5
        found = False

        for attempt in range(n_attempts):
            if check_time():
                break

            np.random.seed(attempt * 1000 + r)
            # Initialize with small random values
            U = np.random.randint(-1, 2, size=(nm, r)).astype(float)
            V = np.random.randint(-1, 2, size=(mp, r)).astype(float)
            W = np.random.randint(-1, 2, size=(pn, r)).astype(float)

            # ALS iterations
            max_iters = 50
            for it in range(max_iters):
                if check_time():
                    break

                # Update U
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed V, W: T[a] = sum_r U[a,r] * (V[:,r] * W[:,r])
                # D_U[b*pn+c, r] = V[b,r] * W[c,r]
                D_U = np.zeros((mp * pn, r))
                for ri in range(r):
                    D_U[:, ri] = (V[:, ri].reshape(mp, 1) * W[:, ri].reshape(1, pn)).reshape(-1)

                try:
                    U_new = np.zeros((nm, r))
                    for a in range(nm):
                        t_a = T[a, :, :].reshape(-1)
                        U_new[a, :] = np.linalg.lstsq(D_U, t_a, rcond=None)[0]
                    U = U_new
                except:
                    break

                # Update V
                D_V = np.zeros((nm * pn, r))
                for ri in range(r):
                    D_V[:, ri] = (U[:, ri].reshape(nm, 1) * W[:, ri].reshape(1, pn)).reshape(-1)

                try:
                    V_new = np.zeros((mp, r))
                    for b in range(mp):
                        t_b = T[:, b, :].reshape(-1)
                        V_new[b, :] = np.linalg.lstsq(D_V, t_b, rcond=None)[0]
                    V = V_new
                except:
                    break

                # Update W
                D_W = np.zeros((nm * mp, r))
                for ri in range(r):
                    D_W[:, ri] = (U[:, ri].reshape(nm, 1) * V[:, ri].reshape(1, mp)).reshape(-1)

                try:
                    W_new = np.zeros((pn, r))
                    for c in range(pn):
                        t_c = T[:, :, c].reshape(-1)
                        W_new[c, :] = np.linalg.lstsq(D_W, t_c, rcond=None)[0]
                    W = W_new
                except:
                    break

            # Check if we found a good solution
            if not check_time():
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2

                if verify_fast(U_r, V_r, W_r):
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_r]
                    V_list = [[float(x) for x in row] for row in V_r]
                    W_list = [[float(x) for x in row] for row in W_r]
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break

                # Also try without rounding (check if already close to integers)
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify_fast(U_int, V_int, W_int):
                    U_list = [[float(x) for x in row] for row in U_int]
                    V_list = [[float(x) for x in row] for row in V_int]
                    W_list = [[float(x) for x in row] for row in W_int]
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break

        if found:
            # Try to go lower
            continue

    # Return best found or fallback
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
