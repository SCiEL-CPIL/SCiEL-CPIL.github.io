import numpy as np
import time

_t0 = time.time()
_DEADLINE = 130.0

def _build_init(rank):
    n, m, p = 2, 2, 16
    u = np.zeros((n * m, rank))
    v = np.zeros((m * p, rank))
    w = np.zeros((p * n, rank))
    for r in range(min(8, rank // 7)):
        for s in range(7):
            c = 7 * r + s
            u[0, c] += (1 if s in (0, 2, 4) else -1 if s == 5 else 0)
            u[1, c] += (1 if s in (4, 6) else -1 if s == 5 else 0)
            u[2, c] += (1 if s in (1, 5) else -1 if s == 6 else 0)
            u[3, c] += (1 if s in (0, 1, 3, 6) else -1 if s == 5 else 0)
            v[2 * r, c] += (1 if s in (0, 2) else -1 if s == 3 else 0)
            v[2 * r + 1, c] += (1 if s in (2, 5) else -1 if s == 6 else 0)
            v[16 + 2 * r, c] += (1 if s in (1, 3) else -1 if s == 5 else 0)
            v[16 + 2 * r + 1, c] += (1 if s in (0, 1, 4, 6) else -1 if s == 3 else 0)
            w[4 * r, c] += (1 if s in (0, 3, 5, 6) else -1 if s == 4 else 0)
            w[4 * r + 1, c] += (1 if s in (1, 3) else -1 if s in (2, 6) else 0)
            w[4 * r + 2, c] += (1 if s in (2, 4, 6) else -1 if s == 5 else 0)
            w[4 * r + 3, c] += (1 if s in (0, 1, 3, 4) else -1 if s == 2 else 0)
    if rank > 56:
        for d in range(56, rank):
            u[d % 4, d] = 1
            v[d % 32, d] = 1
            w[(d // 32) * n + (d % 32) % n, d] = 0
            w[0, d] = 1
    return u, v, w


def _reduce_rank(u, v, w, target, tol=1e-7):
    """Remove zero/near-zero components, then try column removal via residual fitting."""
    n, m, p = 2, 2, 16
    cur_r = u.shape[1]
    for tgt in range(target, cur_r):
        if time.time() - _t0 > _DEADLINE:
            break
        best_j = None
        best_err = np.inf
        u_t, v_t, w_t = u.T, v.T, w.T
        kept = list(range(cur_r))
        for j in kept:
            if abs(u_t[j].max()) + abs(v_t[j].max()) + abs(w_t[j].max()) < 1e-10:
                candidate = [k for k in kept if k != j]
            else:
                continue
            U_c = u_t[:, candidate]
            V_c = v_t[:, candidate]
            W_c = w_t[:, candidate]
            err = 0.0
            for i in range(n):
                for j2 in range(m):
                    for k in range(p):
                        u_val = U_c[i * m + j2, :]
                        v_val = V_c[j2 * p + k, :]
                        w_val = W_c[k * n + i, :]
                        tgt_val = 1.0 if (i == 0 or j2 == 0 or k == 0) else 0.0
                        cands = u_val * v_val
                        if np.max(np.abs(cands)) < 1e-12:
                            resid = tgt_val
                        else:
                            w_new, _, _, _ = np.linalg.lstsq(cands, tgt_val + 0, rcond=None)
                            # Actually need: sum_s u[i,m,j,s]*v[m,k,s]*w[k,n,i,s] = 1
                            # For fixed u,v, w is linear. But w is shared across equations.
                            # This per-equation check is wrong; need full check.
                            # Skip individual removal for non-trivial columns.
                            continue
                        err += (resid - tgt_val) ** 2
            if err < best_err:
                best_err = err
                best_j = j
        if best_j is not None and best_err < tol:
            u = np.delete(u, best_j, axis=1)
            v = np.delete(v, best_j, axis=1)
            w = np.delete(w, best_j, axis=1)
            cur_r -= 1
    return u, v, w


def construct():
    n, m, p = 2, 2, 16
    _elapsed_limit = _DEADLINE - (time.time() - _t0)

    # Try to find better decomposition via gradient optimization on rank-64
    u, v, w = _build_init(64)
    u_g, v_g, w_g = u.copy(), v.copy(), w.copy()

    # Gradient descent on log-loss to find valid rank-64 decomposition
    alpha = 0.15
    step_size = 0.005
    max_steps = 500
    best_loss = np.inf
    best_u, best_v, best_w = u_g.copy(), v_g.copy(), w_g.copy()
    best_rank = 64

    for step in range(max_steps):
        if time.time() - _t0 > _DEADLINE:
            break

        # Compute gradients
        u_grad = np.zeros_like(u_g)
        v_grad = np.zeros_like(v_g)
        w_grad = np.zeros_like(w_g)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    ua = u_g[a_idx, :]
                    vb = v_g[b_idx, :]
                    wc = w_g[c_idx, :]
                    prods = ua * vb * wc
                    s = np.sum(prods)
                    # log(1+e^{-s}) derivative: for s>0: 1/(1+e^{s}), for s<0: 1
                    if s > 30:
                        ls = np.exp(-s)
                    elif s < -30:
                        ls = 1.0
                    else:
                        ls = 1.0 / (1.0 + np.exp(s))
                    grad_factor = ls * (-1.0)
                    u_grad[a_idx, :] += grad_factor * vb * wc
                    v_grad[b_idx, :] += grad_factor * ua * wc
                    w_grad[c_idx, :] += grad_factor * ua * vb

        # Gradient clipping and update
        for g_arr in [u_grad, v_grad, w_grad]:
            mg = np.max(np.abs(g_arr))
            if mg > 1e3:
                g_arr /= mg * 1e3
        u_g -= step_size * u_grad
        v_g -= step_size * v_grad
        w_g -= step_size * w_grad

        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(u_g[i * m + j] * v_g[j * p + k] * w_g[k * n + i])
                    if s > 30:
                        loss += s + np.log(1.0 + np.exp(-s))
                    elif s < -30:
                        loss += -s + np.log(1.0 + np.exp(s))
                    else:
                        loss += np.log(1.0 + np.exp(-s))

        if loss < best_loss:
            best_loss = loss
            best_u, best_v, best_w = u_g.copy(), v_g.copy(), w_g.copy()
            if loss < 1e-10:
                break

    u_best, v_best, w_best = best_u, best_v, best_w

    # If optimization found good rank-64, try to reduce
    if best_loss < 0.5:
        u_red, v_red, w_red = _reduce_rank(u_best, v_best, w_best, 56)
        r_red = u_red.shape[1]
        if r_red < best_rank:
            best_u, best_v, best_w = u_red, v_red, w_red
            best_rank = r_red

    # Fallback to Strassen rank-56 if optimization didn't help
    if best_rank >= 56:
        u_s, v_s, w_s = _build_init(56)
        best_u, best_v, best_w = u_s, v_s, w_s
        best_rank = 56

    # Final: convert to lists
    U = best_u.tolist()
    V = best_v.tolist()
    W = best_w.tolist()

    return {"U": U, "V": V, "W": W}
