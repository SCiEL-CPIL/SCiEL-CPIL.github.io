"""Rewritten program for rank-minimal tensor decomposition of 2x5 by 5x5 matrix multiplication."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 150s wall-clock limit

    # Build the target tensor T: T[i,j,k] = 1 if (i,m,j,p,k) is a valid multiplication
    # Using the convention: input indices (i,m) -> a=i*m+j, (j,p) -> b=j*p+k, (k,n) -> c=k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))

    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for a, b, c in target_indices:
        T_target[a, b, c] = 1.0

    # Precompute some useful dimensions
    dim_U = n * m  # 10
    dim_V = m * p  # 25
    dim_W = p * n  # 10

    # Function to check validity and snap to half-integers
    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Snap to nearest half-integer
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2

        # Reconstruct tensor
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)

        # Check exactness
        if not np.allclose(T_recon, T_target, atol=1e-8):
            return None

        # Convert back to lists
        U_out = [[float(x) for x in row] for row in U_snap]
        V_out = [[float(x) for x in row] for row in V_snap]
        W_out = [[float(x) for x in row] for row in W_snap]

        return {"U": U_out, "V": V_out, "W": W_out}

    # Optimization function for a given rank
    def try_rank(r, seed=42):
        if time.time() > deadline:
            return None

        np.random.seed(seed)

        # Initialize U, V, W randomly
        U = np.random.randn(dim_U, r) * 0.3
        V = np.random.randn(dim_V, r) * 0.3
        W = np.random.randn(dim_W, r) * 0.3

        size_U = dim_U * r
        size_V = dim_V * r
        size_W = dim_W * r

        def loss(params):
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]

            U_p = U_flat.reshape((dim_U), r)
            V_p = V_flat.reshape((dim_V), r)
            W_p = W_flat.reshape((dim_W), r)

            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err

        def grad(params):
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]

            U_p = U_flat.reshape((dim_U), r)
            V_p = V_flat.reshape((dim_V), r)
            W_p = W_flat.reshape((dim_W), r)

            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            diff = T_recon - T_target

            # Gradient of sum(diff^2) w.r.t. U[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V_p, W_p)
            grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U_p, W_p)
            grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U_p, V_p)

            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

        try:
            # Use L-BFGS-B with gradient for faster convergence
            res = minimize(loss, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            return None

        # Check if the solution is good enough
        if res.fun < 1e-4:
            U_sol = res.x[:size_U].reshape((dim_U), r)
            V_sol = res.x[size_U:size_U + size_V].reshape((dim_V), r)
            W_sol = res.x[size_U + size_V:].reshape((dim_W), r)

            result = check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
            if result:
                return result
        return None

    # Fallback schoolbook decomposition
    rank_school = n * m * p  # 50
    U_school = [[0.0] * rank_school for _ in range(dim_U)]
    V_school = [[0.0] * rank_school for _ in range(dim_V)]
    W_school = [[0.0] * rank_school for _ in range(dim_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find lower rank decompositions
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25
    # Known results suggest rank is between 25 and 50 for this case
    # Try ranks from just above the lower bound up to just below schoolbook
    ranks_to_try = list(range(26, 49))  # 26 to 48

    # Sort by likelihood of success: try moderate reductions first
    # Actually, let's try from high to low, stopping when we find one
    ranks_to_try = [48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26]

    seeds = [42, 123, 999, 111, 222, 333, 444, 555, 666, 777, 888, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
             11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999, 12345]

    for r in ranks_to_try:
        if time.time() > deadline:
            break
        found = False
        for seed in seeds:
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result:
                best_result = result
                best_rank = r
                found = True
                break
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
