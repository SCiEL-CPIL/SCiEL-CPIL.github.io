"""Matrix multiplication 2x4 x 4x6 rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Schoolbook fallback: rank 48
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    def decompose_residual(U, V, W, rank):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T_dec = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.linalg.norm(T_dec - np.ones((n, m, p)))

    def verify_exact(U, V, W, rank, denom):
        Uf = [[Fraction(int(round(U[i, c] * denom)), denom) for c in range(rank)] for i in range(nm)]
        Vf = [[Fraction(int(round(V[i, c] * denom)), denom) for c in range(rank)] for i in range(mp)]
        Wf = [[Fraction(int(round(W[i, c] * denom)), denom) for c in range(rank)] for i in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def try_rank_opt(rank, n_restarts=3, max_iters=2000):
        nonlocal best_U, best_V, best_W, best_rank
        for restart in range(n_restarts):
            if time.time() > DEADLINE:
                return False

            rng = np.random.RandomState(restart * 7919 + rank * 104729)
            scale = 1.0 / np.sqrt(rank)
            U = rng.randn(nm, rank) * scale
            V = rng.randn(mp, rank) * scale
            W = rng.randn(pn, rank) * scale

            lr = 0.05
            best_res = decompose_residual(U, V, W, rank)

            for it in range(max_iters):
                if time.time() > DEADLINE:
                    return False

                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                T_dec = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                residual = T_dec - 1.0
                cur_res = np.linalg.norm(residual)

                if cur_res < 1e-7:
                    break

                dU = 2 * np.einsum('ijk,jkc,kic->ijc', residual, V3, W3)
                dV = 2 * np.einsum('ijk,ijc,kic->jkc', residual, U3, W3)
                dW = 2 * np.einsum('ijk,ijc,jkc->kic', residual, U3, V3)

                gn = np.sqrt(np.sum(dU ** 2) + np.sum(dV ** 2) + np.sum(dW ** 2))
                if gn < 1e-12:
                    break

                step = lr / max(1.0, gn * 0.1)
                U_new = U - step * dU
                V_new = V - step * dV
                W_new = W - step * dW

                new_res = decompose_residual(U_new, V_new, W_new, rank)
                if new_res < cur_res:
                    U, V, W = U_new, V_new, W_new
                    best_res = new_res
                else:
                    lr *= 0.5
                    if lr < 1e-10:
                        break

            if best_res < 1e-6:
                # Try to rationalize
                for denom in [1, 2, 3, 4, 6, 12, 24]:
                    U_r = np.round(U * denom) / denom
                    V_r = np.round(V * denom) / denom
                    W_r = np.round(W * denom) / denom
                    T_dec = np.einsum('ijc,jkc,kic->ijk',
                                      U_r.reshape(n, m, rank), V_r.reshape(m, p, rank), W_r.reshape(p, n, rank))
                    if np.allclose(T_dec, 1.0, atol=1e-9):
                        if verify_exact(U_r, V_r, W_r, rank, denom):
                            best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rank
                            return True

        return False

    # Try ranks from lower bound upward
    for rank in range(24, 42):
        if time.time() > DEADLINE - 10:
            break
        if try_rank_opt(rank):
            break

    # Convert to list format
    r = best_rank
    U_list = [[float(best_U[i, c]) for c in range(r)] for i in range(nm)]
    V_list = [[float(best_V[i, c]) for c in range(r)] for i in range(mp)]
    W_list = [[float(best_W[i, c]) for c in range(r)] for i in range(pn)]

    return {"U": U_list, "V": V_list, "W": W_list}
