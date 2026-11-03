"""
Rank-minimal tensor decomposition for 3x3 * 3x7 matrix multiplication.
Strategy: Use the tensor product structure T = T_3x3x3 ⊗ T_3x7,
then apply numerical rank minimization with a structured initialization
that exploits the shared input structure and a global linear transformation
to mix output indices.
"""
import time
import numpy as np
import random
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    hard_limit = 115.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 63

    def time_left():
        return hard_limit - (time.time() - start_time)

    def compute_tensor(U, V, W, r):
        """Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]"""
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return T

    def compute_tensor_and_grad(params, r):
        """
        Computes the tensor T and the gradient of the squared error ||T - 1||^2.
        params: flattened array [U_flat, V_flat, W_flat]
        """
        U = params[0:r*n*m].reshape(n * m, r)
        V = params[r*n*m:2*r*n*m].reshape(m * p, r)
        W = params[2*r*n*m:3*r*n*m].reshape(p * n, r)

        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)

        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = T - 1.0
        obj = np.sum(err * err)

        # Gradients
        grad_U_3d = np.einsum('ijk,jkc,kic->ijc', err, V_3d, W_3d)
        grad_V_3d = np.einsum('ijk,ijc,kic->jkc', err, U_3d, W_3d)
        grad_W_3d = np.einsum('ijk,ijc,jkc->kic', err, U_3d, V_3d)

        grad = 2.0 * np.concatenate([
            grad_U_3d.flatten(),
            grad_V_3d.flatten(),
            grad_W_3d.flatten()
        ])

        return obj, grad

    def objective_only(params, r):
        obj, _ = compute_tensor_and_grad(params, r)
        return obj

    def verify_solution(U, V, W, r):
        """Check if the solution is exact (within tolerance)"""
        T = compute_tensor(U, V, W, r)
        return np.allclose(T, 1.0, atol=1e-9)

    def snap_to_half_integers(U, V, W, r):
        """Try to snap to half-integer values and verify"""
        # Try snapping to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2

        if verify_solution(U_snap, V_snap, W_snap, r):
            return U_snap, V_snap, W_snap, True

        # Try snapping to integers
        U_snap = np.round(U)
        V_snap = np.round(V)
        W_snap = np.round(W)

        if verify_solution(U_snap, V_snap, W_snap, r):
            return U_snap, V_snap, W_snap, True

        # Try mixed: some half, some integer
        for shift in [0, 0.25, 0.5, 0.75]:
            U_snap = np.round((U - shift) * 2) / 2 + shift
            V_snap = np.round((V - shift) * 2) / 2 + shift
            W_snap = np.round((W - shift) * 2) / 2 + shift
            if verify_solution(U_snap, V_snap, W_snap, r):
                return U_snap, V_snap, W_snap, True

        return U, V, W, False

    def apply_linear_mix(U, V, W, r, transform_type='output'):
        """Apply a global linear transformation to mix output indices"""
        # Try mixing the output (k*n+i) indices
        if transform_type == 'output':
            # Random orthogonal transformation in the output space
            size = p * n
            Q = np.eye(size)
            # Apply a few random Givens rotations
            for _ in range(3):
                i = random.randint(0, size - 2)
                j = random.randint(i + 1, size - 1)
                theta = random.uniform(-1, 1)
                c, s = np.cos(theta), np.sin(theta)
                G = np.eye(size)
                G[i, i] = c
                G[i, j] = -s
                G[j, i] = s
                G[j, j] = c
                Q = G @ Q
            # Apply to W
            W_new = W @ Q
            return U, V, W_new, r

        return U, V, W, r

    def optimize_with_lbfgs(U0, V0, W0, r, maxiter=1000):
        """Optimize using L-BFGS-B"""
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

        try:
            res = minimize(
                lambda x: objective_only(x, r),
                params0,
                jac=lambda x: compute_tensor_and_grad(x, r)[1],
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )

            params_opt = res.x
            U_opt = params_opt[0:r*n*m].reshape(n*m, r)
            V_opt = params_opt[r*n*m:2*r*n*m].reshape(m*p, r)
            W_opt = params_opt[2*r*n*m:3*r*n*m].reshape(p*n, r)

            obj_val = objective_only(params_opt, r)
            return U_opt, V_opt, W_opt, obj_val

        except Exception:
            return U0, V0, W0, float('inf')

    def optimize_with_trust_ncg(U0, V0, W0, r, maxiter=500):
        """Optimize using trust-ncg (handles constraints better)"""
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

        try:
            res = minimize(
                lambda x: objective_only(x, r),
                params0,
                jac=lambda x: compute_tensor_and_grad(x, r)[1],
                method='trust-ncg',
                options={'maxiter': maxiter, 'gtol': 1e-8}
            )

            params_opt = res.x
            U_opt = params_opt[0:r*n*m].reshape(n*m, r)
            V_opt = params_opt[r*n*m:2*r*n*m].reshape(m*p, r)
            W_opt = params_opt[2*r*n*m:3*r*n*m].reshape(p*n, r)

            obj_val = objective_only(params_opt, r)
            return U_opt, V_opt, W_opt, obj_val

        except Exception:
            return U0, V0, W0, float('inf')

    # Schoolbook Solution
    U_school = np.zeros((n*m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, c] = 1
                V_school[j*p+k, c] = 1
                W_school[k*n+i, c] = 1
                c += 1

    best_rank = rank_schoolbook
    best_solution = (U_school.copy(), V_school.copy(), W_school.copy())
    best_obj = float('inf')

    # Tensor product structure: T_3x3x3 has rank 21 (Landsberg bound)
    # T_3x7 is diagonal with rank 7
    # Tensor product rank is at most 21 * 7 = 147, but we can do better
    # Actually, for this specific case, let's use a structured approach

    # Structured initialization using tensor product
    def structured_tensor_product_init(r, seed=None):
        """Initialize using tensor product structure with random mixing"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Base 3x3x3 multiplication tensor (rank 21)
        # Use a known decomposition for 3x3x3
        U_333 = np.zeros((9, 21))
        V_333 = np.zeros((9, 21))
        W_333 = np.zeros((9, 21))

        # Simple structured decomposition for 3x3x3
        c = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_333[i*3+j, c] = 1
                    V_333[j*3+k, c] = 1
                    W_333[k*3+i, c] = 1
                    c += 1

        # For 3x7, we can use a rank-7 decomposition
        # T_3x7[i,j,k] = delta[i,j] * delta[j,k] for j in 0..2, k in 0..6
        # This is just the identity on the 3x3 part, extended to 3x7
        U_37 = np.zeros((9, 7))
        V_37 = np.zeros((21, 7))
        W_37 = np.zeros((21, 7))

        for j in range(3):
            for k in range(7):
                U_37[j*3 + min(j, 2), j*7 + k] = 1 if j < 3 else 0
                V_37[j*7 + k, j*7 + k] = 1
                W_37[k*3 + j, j*7 + k] = 1

        # Combine via tensor product and then mix
        # Total rank needed: min(r, 21 * 7)
        # Use random linear combinations to reduce rank
        U_comb = np.zeros((9, r))
        V_comb = np.zeros((21, r))
        W_comb = np.zeros((21, r))

        for c_idx in range(min(r, 21 * 7)):
            i3 = c_idx % 21
            i7 = c_idx // 21

            if i3 < 21 and i7 < 7:
                U_comb[:, c_idx] = U_333[:, i3] * U_37[:, i7]
                V_comb[:, c_idx] = V_333[:, i3] * V_37[:, i7]
                W_comb[:, c_idx] = W_333[:, i3] * W_37[:, i7]

        # Apply random mixing to reduce effective rank
        # Use SVD-like compression
        if r < 21 * 7:
            # Random projection
            P = np.random.randn(21 * 7, r)
            U_comb = U_comb @ P[:21, :]
            V_comb = V_comb @ P[:21, :]
            W_comb = W_comb @ P[:21, :]

        return U_comb, V_comb, W_comb

    # Try different ranks from high to low
    # Known: schoolbook is 63, let's try to go below that
    min_rank_to_try = 35
    max_rank_to_try = rank_schoolbook - 1

    # Optimization strategies
    for r in range(max_rank_to_try, min_rank_to_try - 1, -1):
        if time_left() < 5:
            break

        found_valid = False
        num_restarts = 4 if r > 50 else 6

        for trial in range(num_restarts):
            if time_left() < 3:
                break

            seed = trial * 1000 + r * 7
            random.seed(seed)
            np.random.seed(seed)

            # Different initialization strategies
            if trial == 0:
                # Random small values + bias
                U0 = np.random.randn(n*m, r) * 0.1 + 0.5
                V0 = np.random.randn(m*p, r) * 0.1 + 0.5
                W0 = np.random.randn(p*n, r) * 0.1 + 0.5
            elif trial == 1:
                # Structured tensor product init
                U0, V0, W0 = structured_tensor_product_init(r, seed)
                # Apply output mixing
                U0, V0, W0, r = apply_linear_mix(U0, V0, W0, r, 'output')
            elif trial == 2:
                # Sparse initialization
                U0 = np.zeros((n*m, r))
                V0 = np.zeros((m*p, r))
                W0 = np.zeros((p*n, r))
                for i in range(n*m):
                    col = random.randint(0, r-1)
                    U0[i, col] = random.choice([0.5, 1.0])
                for j in range(m*p):
                    col = random.randint(0, r-1)
                    V0[j, col] = random.choice([0.5, 1.0])
                for k in range(p*n):
                    col = random.randint(0, r-1)
                    W0[k, col] = random.choice([0.5, 1.0])
            elif trial == 3:
                # Based on schoolbook pattern with perturbation
                U0 = np.zeros((n*m, r))
                V0 = np.zeros((m*p, r))
                W0 = np.zeros((p*n, r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = (i * m * p + j * p + k) % r
                            U0[i*m+j, c] += 1.0
                            V0[j*p+k, c] += 1.0
                            W0[k*n+i, c] += 1.0
                # Add small random perturbation
                U0 += np.random.randn(n*m, r) * 0.01
                V0 += np.random.randn(m*p, r) * 0.01
                W0 += np.random.randn(p*n, r) * 0.01
            else:
                # Very small random values with structure
                U0 = np.random.randn(n*m, r) * 0.05
                V0 = np.random.randn(m*p, r) * 0.05
                W0 = np.random.randn(p*n, r) * 0.05
                # Ensure some non-zero structure
                for i in range(n*m):
                    U0[i, i % r] += 0.1
                for j in range(m*p):
                    V0[j, j % r] += 0.1
                for k in range(p*n):
                    W0[k, k % r] += 0.1

            # Try L-BFGS first
            U_opt, V_opt, W_opt, obj_val = optimize_with_lbfgs(U0, V0, W0, r, maxiter=800)

            # Check if we found a very good solution
            if obj_val < 1e-6:
                U_snap, V_snap, W_snap, is_valid = snap_to_half_integers(U_opt, V_opt, W_opt, r)

                if is_valid:
                    best_rank = r
                    best_solution = (U_snap, V_snap, W_snap)
                    found_valid = True
                    break
            elif obj_val < 1e-3:
                # If we're close but not quite there, try trust-ncg for polishing
                if time_left() > 10:
                    U_opt2, V_opt2, W_opt2, obj_val2 = optimize_with_trust_ncg(
                        U_opt, V_opt, W_opt, r, maxiter=300
                    )
                    if obj_val2 < 1e-6:
                        U_snap, V_snap, W_snap, is_valid = snap_to_half_integers(
                            U_opt2, V_opt2, W_opt2, r
                        )
                        if is_valid:
                            best_rank = r
                            best_solution = (U_snap, V_snap, W_snap)
                            found_valid = True
                            break
            # If obj_val is between 1e-3 and 1e-6, try one more L-BFGS with different init
            elif obj_val < 1e-2 and time_left() > 8:
                # Perturb the current solution and retry
                U0b = U_opt + np.random.randn(n*m, r) * 0.001
                V0b = V_opt + np.random.randn(m*p, r) * 0.001
                W0b = W_opt + np.random.randn(p*n, r) * 0.001
                U_opt, V_opt, W_opt, obj_val = optimize_with_lbfgs(U0b, V0b, W0b, r, maxiter=500)
                if obj_val < 1e-6:
                    U_snap, V_snap, W_snap, is_valid = snap_to_half_integers(U_opt, V_opt, W_opt, r)
                    if is_valid:
                        best_rank = r
                        best_solution = (U_snap, V_snap, W_snap)
                        found_valid = True
                        break

            # If we haven't found a valid solution, update best if this is better
            if obj_val < best_obj:
                best_obj = obj_val

        if found_valid:
            continue  # Try next lower rank

    U_res, V_res, W_res = best_solution

    def to_list(arr):
        return arr.tolist()

    return {"U": to_list(U_res), "V": to_list(V_res), "W": to_list(W_res)}
