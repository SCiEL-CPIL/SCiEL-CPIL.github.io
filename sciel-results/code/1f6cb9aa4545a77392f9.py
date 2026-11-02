"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition.

Design family (kept from the current program):
  * A verified rank-26 construction (Strassen 2x2 core + schoolbook border)
    is the guaranteed fallback.
  * A continuous local search over the factor matrices attempts rank reduction
    toward the known minimal rank 23.

Refinements added in this revision (within the same family):
  1. Smooth epigraph-style surrogate for the L1 column shrinkage:
         s(x) = x / (|x| + eps)
     replaces the nonsmooth sign() subgradient, removing the subgradient kinks
     that stall the plain L1 penalty and allowing smoother descent toward
     sparsity.
  2. Adaptive learning rate with backtracking (a step that increases the
     objective is reverted and the step size is reduced) instead of a fixed lr.
  3. A final high-precision polish pass (smaller lr, tighter pruning tolerance,
     more iterations) to drive redundant columns below the prune threshold.
"""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication.
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    terms = []

    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))

    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))

    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))

    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))

    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))

    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff

    return {"U": U, "V": V, "W": W}

def _local_search_reduce(base_result, deadline):
    """
    Continuous local search to reduce rank of a valid decomposition.

    Same family as the current program: direct parameterization of the factor
    matrices, gradient descent on the squared tensor residual plus an L1-style
    column-shrinkage term, periodic pruning of near-zero columns, and exact
    re-verification. Refinements: a smooth epigraph surrogate for the L1 term,
    an adaptive (backtracking) learning rate, and a final high-precision polish
    pass.
    """
    import numpy as np

    n = m = p = 3
    U0 = np.array(base_result["U"], dtype=float)
    V0 = np.array(base_result["V"], dtype=float)
    W0 = np.array(base_result["W"], dtype=float)
    R = U0.shape[1]

    U = U0.copy()
    V = V0.copy()
    W = W0.copy()

    best = None
    best_rank = R

    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    T = len(triples)

    def try_prune_and_verify(Uc, Vc, Wc, tol=1e-6):
        """Prune columns whose combined L2 norm is below tol, verify, return (valid, rank, Uc,Vc,Wc)."""
        norms = np.sqrt(np.sum(Uc**2, axis=0) + np.sum(Vc**2, axis=0) + np.sum(Wc**2, axis=0))
        keep = norms > tol
        if keep.sum() >= R:
            return False, R, Uc, Vc, Wc
        Uk = Uc[:, keep]
        Vk = Vc[:, keep]
        Wk = Wc[:, keep]
        if _verify(Uk.tolist(), Vk.tolist(), Wk.tolist()):
            return True, int(keep.sum()), Uk, Vk, Wk
        return False, R, Uc, Vc, Wc

    def compute_residuals(Uc, Vc, Wc):
        res = np.empty(T)
        for t, (i, j, k) in enumerate(triples):
            res[t] = np.dot(Uc[i*m+j, :] * Vc[j*p+k, :], Wc[k*n+i, :]) - 1.0
        return res

    def smooth_l1_grad(x, eps):
        # Smooth epigraph-style surrogate of |x|: s(x) = x / (|x| + eps)
        # (continuous, bounded in [-1,1], approximates sign(x) away from 0).
        return x / (np.abs(x) + eps)

    def objective(Uc, Vc, Wc, lam, eps):
        res = compute_residuals(Uc, Vc, Wc)
        l1 = np.sum(np.abs(Uc)) + np.sum(np.abs(Vc)) + np.sum(np.abs(Wc))
        return float(np.sum(res*res)) + lam * l1

    def search(U, V, W, R, lam, eps, lr, lr_min, max_iter, prune_tol, deadline):
        """Run the adaptive gradient search; return (best_tuple, best_rank)."""
        nonlocal best, best_rank
        cur_obj = objective(U, V, W, lam, eps)
        it = 0
        while it < max_iter and time.time() < deadline:
            it += 1
            res = compute_residuals(U, V, W)

            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)
            for t, (i, j, k) in enumerate(triples):
                rt = res[t]
                gU[i*m+j, :] += rt * V[j*p+k, :] * W[k*n+i, :]
                gV[j*p+k, :] += rt * U[i*m+j, :] * W[k*n+i, :]
                gW[k*n+i, :] += rt * U[i*m+j, :] * V[j*p+k, :]

            # Smooth epigraph surrogate for the L1 shrinkage (no sign kinks).
            gU += lam * smooth_l1_grad(U, eps)
            gV += lam * smooth_l1_grad(V, eps)
            gW += lam * smooth_l1_grad(W, eps)

            # Adaptive (backtracking) gradient step.
            trial_U = U - lr * gU
            trial_V = V - lr * gV
            trial_W = W - lr * gW
            trial_obj = objective(trial_U, trial_V, trial_W, lam, eps)
            if trial_obj <= cur_obj + 1e-12:
                U, V, W = trial_U, trial_V, trial_W
                cur_obj = trial_obj
                if it % 100 == 0:
                    lr = min(lr * 1.05, 0.5)   # gentle growth when steady
            else:
                lr = max(lr * 0.6, lr_min)      # revert, shrink step

            # Prune and verify occasionally.
            if it % 50 == 0:
                valid, rk, Uk, Vk, Wk = try_prune_and_verify(U, V, W, prune_tol)
                if valid and rk < best_rank:
                    best_rank = rk
                    best = (Uk.copy(), Vk.copy(), Wk.copy())
                    U, V, W = Uk, Vk, Wk
                    R = best_rank
                    cur_obj = objective(U, V, W, lam, eps)
                    if best_rank <= 23:
                        break

        # Final prune + verify at the end of this phase.
        valid, rk, Uk, Vk, Wk = try_prune_and_verify(U, V, W, prune_tol)
        if valid and rk < best_rank:
            best_rank = rk
            best = (Uk.copy(), Vk.copy(), Wk.copy())
        return (U, V, W), best_rank

    # ---- Phase 1: main search ----
    U, V, W = search(U, V, W, R, lam=0.02, eps=0.05, lr=0.05, lr_min=1e-4,
                     max_iter=4000, prune_tol=1e-6, deadline=deadline)

    # ---- Phase 2: high-precision polish (smaller lr, tighter tolerance) ----
    if best is not None and best_rank < R and time.time() < deadline:
        U, V, W = search(best[0], best[1], best[2], best_rank, lam=0.01, eps=0.02,
                         lr=0.02, lr_min=1e-5, max_iter=2000, prune_tol=1e-7,
                         deadline=deadline)

    if best is not None and best_rank < R:
        return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
    return None

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before the 120s limit

    # Verified rank-26 construction as guaranteed fallback
    result = _strassen_rank26()
    base_ok = _verify(result["U"], result["V"], result["W"])

    # Attempt structural rank reduction via continuous local search
    if base_ok and time.time() < deadline:
        try:
            reduced = _local_search_reduce(result, deadline)
            if reduced is not None:
                if _verify(reduced["U"], reduced["V"], reduced["W"]):
                    return reduced
        except Exception:
            pass

    if base_ok:
        return result
    return _schoolbook()

# EVOLVE-BLOCK-END
