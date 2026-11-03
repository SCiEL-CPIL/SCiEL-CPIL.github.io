"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50

    # Build target tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def check_and_snap(U, V, W, r):
        """Snap to half-integers and verify exact reconstruction."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None
        return {
            "U": [[float(x) for x in row] for row in U_snap],
            "V": [[float(x) for x in row] for row in V_snap],
            "W": [[float(x) for x in row] for row in W_snap],
        }

    # ---- Structured decomposition via 2+2+1 split of the middle index j ----
    # For a block of j-indices J (size s), the sub-tensor is:
    #   C[i,k] = sum_{j in J} A[i,j] * B[j,k]
    # We decompose each such block. For s=1 (2x1*1x5): rank 10 (each (i,k) is one product).
    # For s=2 (2x2*2x5): we use Strassen on the 2x2 structure of A and B's rows.
    #
    # Strassen for 2x2 * 2x2 gives rank 7. For 2x2 * 2x5, B has 5 columns.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # We can apply a rank-4 decomposition per column-pair but B columns are shared.
    # Simplest verified: treat as 5 independent 2x2*2x1 problems -> rank 5*4 = 20.
    # To beat that, use the 2x2 Strassen structure across the 2 rows of A and 2 rows of B,
    # applied per output column k. For fixed k, B[:,k] is a 2-vector. The product A (2x2) * B[:,k] (2x1)
    # is 2x2 * 2x1 which has rank 4 (schoolbook, tight for this size).
    # So 2x2*2x5 block rank = 5 * 4 = 20.
    #
    # Total: 20 (j=0,1) + 20 (j=2,3) + 10 (j=4) = 50. Same as schoolbook - no gain.
    #
    # To actually reduce, exploit n=2: combine the two i-rows.
    # The full tensor T[(i,j),(j',k),(k',i')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}.
    # We can write T = sum_j (sum_i e_i^A e_i^W) kron-like ... 
    #
    # Given the difficulty of a guaranteed sub-50 exact decomposition within time,
    # we attempt an optimization search for rank 40, then fall back to schoolbook.

    def try_rank(r, seed=42):
        if time.time() > deadline:
            return None
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5

        def loss(params):
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_p = params[:size_U].reshape((n * m), r)
            V_p = params[size_U:size_U + size_V].reshape((m * p), r)
            W_p = params[size_U + size_V:].reshape((p * n), r)
            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            return np.sum((T_recon - T_target) ** 2)

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        from scipy.optimize import minimize
        try:
            res = minimize(loss, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 800, 'ftol': 1e-10})
        except Exception:
            return None
        if res.fun < 1e-6:
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_sol = res.x[:size_U].reshape((n * m), r)
            V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
            W_sol = res.x[size_U + size_V:].reshape((p * n), r)
            return check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
        return None

    # Fallback: schoolbook (guaranteed correct, rank 50)
    rank_school = schoolbook_rank
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Attempt lower ranks
    for r in [40, 38, 35]:
        if time.time() > deadline:
            break
        for seed in [42, 123, 999, 7]:
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result is not None:
                best_result = result
                best_rank = r
                break
        if best_rank < rank_school:
            break

    return best_result
# EVOLVE-BLOCK-END
