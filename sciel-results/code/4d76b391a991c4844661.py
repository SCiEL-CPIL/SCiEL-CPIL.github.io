"""
Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition.

Strategy:
  1. Start with the schoolbook (rank 72) decomposition.
  2. Use a structured iterative method:
     - Parameterize the decomposition with real-valued factors.
     - Use alternating least-squares / gradient-based refinement.
     - After convergence to a real solution, snap to the nearest half-integer grid.
     - Verify exactness on the half-integer lattice.
  3. Additionally, try a "structured" initialization exploiting the known
     algebraic form of matrix multiplication to guide the search.
  4. If no lower-rank half-integer solution is found, fall back to schoolbook.

The evaluator checks:
  For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
"""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    rank_schoolbook = n * m * p  # 72

    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    # ---------- Schoolbook decomposition ----------
    def schoolbook():
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = rank_schoolbook

    # ---------- Verification ----------
    def verify(U_r, V_r, W_r, tol=1e-8):
        r = U_r.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # ---------- Snap to half-integers and verify ----------
    def snap_and_verify(U_r, V_r, W_r):
        """Try snapping to half-integers and verify."""
        for scale in [2.0, 1.0]:
            cU = np.round(U_r * scale) / scale
            cV = np.round(V_r * scale) / scale
            cW = np.round(W_r * scale) / scale
            if verify(cU, cV, cW, tol=1e-6):
                return cU, cV, cW
        return None

    # ---------- Gradient-based search for a given rank ----------
    def gradient_search(target_rank, n_iter=300, lr_init=0.02):
        """
        Find a real-valued decomposition of the multiplication tensor
        at the given rank using gradient descent with adaptive learning rate.
        """
        if time.time() - start_time > deadline:
            return None

        np.random.seed(42)
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5

        lr = lr_init
        prev_loss = np.inf
        stall_count = 0

        for it in range(n_iter):
            if time.time() - start_time > deadline:
                return None

            # Compute loss and gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            loss = 0.0

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = U[i * m + j, :]
                        v = V[j * p + k, :]
                        w = W[k * n + i, :]
                        val = np.dot(u, v * w)
                        err = val - 1.0
                        loss += err * err

                        # d/dU: err * v * w
                        grad_U[i * m + j, :] += err * v * w
                        # d/dV: err * u * w
                        grad_V[j * p + k, :] += err * u * w
                        # d/dW: err * u * v
                        grad_W[k * n + i, :] += err * u * v

            # Adaptive learning rate
            if loss < prev_loss * 0.95:
                stall_count = 0
                lr = min(lr * 1.1, 0.1)
            else:
                stall_count += 1
                lr *= 0.8

            prev_loss = loss

            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

            # Early stopping
            if loss < 1e-10:
                break
            if stall_count > 50:
                break

        if loss > 1e-6:
            return None

        # Try to snap
        result = snap_and_verify(U, V, W)
        return result

    # ---------- Alternating least squares approach ----------
    def als_search(target_rank, n_iter=200):
        """
        Alternating least squares: fix two factors, solve for the third.
        """
        if time.time() - start_time > deadline:
            return None

        np.random.seed(123)
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5

        for it in range(n_iter):
            if time.time() - start_time > deadline:
                return None

            # Update U: for each (i,j), solve min sum_k (1 - sum_r V[j*p+k,r]*W[k*n+i,r]*U[i*m+j,r])^2
            # This is a linear system in U[i*m+j, :]
            # Let A_{(i,j),r} = sum_k V[j*p+k,r] * W[k*n+i,r]
            # Then we need A_{(i,j),:} . U[i*m+j,:] = 1
            # If target_rank > 1, this is underdetermined; use least squares.

            for i in range(n):
                for j in range(m):
                    # Build the row of the "A" matrix for this (i,j)
                    a_row = np.zeros(target_rank)
                    for k in range(p):
                        a_row += V[j * p + k, :] * W[k * n + i, :]
                    # Solve a_row . x = 1 in least-squares sense
                    # x = a_row^T / (a_row . a_row) if a_row != 0
                    norm_sq = np.dot(a_row, a_row)
                    if norm_sq > 1e-12:
                        U[i * m + j, :] = a_row / norm_sq
                    else:
                        # Fallback: keep current value
                        pass

            # Similarly update V
            for j in range(m):
                for k in range(p):
                    a_row = np.zeros(target_rank)
                    for i in range(n):
                        a_row += U[i * m + j, :] * W[k * n + i, :]
                    norm_sq = np.dot(a_row, a_row)
                    if norm_sq > 1e-12:
                        V[j * p + k, :] = a_row / norm_sq

            # Similarly update W
            for k in range(p):
                for i in range(n):
                    a_row = np.zeros(target_rank)
                    for j in range(m):
                        a_row += U[i * m + j, :] * V[j * p + k, :]
                    norm_sq = np.dot(a_row, a_row)
                    if norm_sq > 1e-12:
                        W[k * n + i, :] = a_row / norm_sq

            # Check loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        loss += (val - 1.0) ** 2

            if loss < 1e-10:
                break

        if loss > 1e-6:
            return None

        result = snap_and_verify(U, V, W)
        return result

    # ---------- Try structured initialization ----------
    def structured_init_search(target_rank, n_iter=200):
        """
        Use a structured initialization based on the known form of matrix multiplication
        to guide the search.
        """
        if time.time() - start_time > deadline:
            return None

        # Initialize with a structured pattern:
        # For rank r, assign each (i,j,k) triple to a column as evenly as possible
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))

        triples = list(product(range(n), range(m), range(p)))
        np.random.seed(7)
        np.random.shuffle(triples)

        for idx, (i, j, k) in enumerate(triples):
            col = idx % target_rank
            U[i * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + i, col] = 1.0

        # Now refine with gradient descent
        lr = 0.01
        prev_loss = np.inf

        for it in range(n_iter):
            if time.time() - start_time > deadline:
                return None

            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            loss = 0.0

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = U[i * m + j, :]
                        v = V[j * p + k, :]
                        w = W[k * n + i, :]
                        val = np.dot(u, v * w)
                        err = val - 1.0
                        loss += err * err
                        grad_U[i * m + j, :] += err * v * w
                        grad_V[j * p + k, :] += err * u * w
                        grad_W[k * n + i, :] += err * u * v

            if loss < prev_loss * 0.95:
                lr = min(lr * 1.05, 0.05)
            else:
                lr *= 0.9

            prev_loss = loss
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

            if loss < 1e-10:
                break

        if loss > 1e-6:
            return None

        result = snap_and_verify(U, V, W)
        return result

    # ---------- Main search loop ----------
    # Try ranks from 71 down to 60
    for target_rank in range(71, 59, -1):
        if time.time() - start_time > deadline:
            break

        # Try multiple methods
        for method in [gradient_search, als_search, structured_init_search]:
            if time.time() - start_time > deadline:
                break
            result = method(target_rank)
            if result is not None:
                cU, cV, cW = result
                best_U, best_V, best_W = cU, cV, cW
                best_rank = target_rank
                break

        if best_rank < rank_schoolbook:
            break

    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-6):
        # Fall back to schoolbook
        best_U, best_V, best_W = schoolbook()
        best_rank = rank_schoolbook

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
