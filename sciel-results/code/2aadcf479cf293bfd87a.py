"""ODE solver for 15-dim stiff linear system using scaling-and-squaring with Taylor series."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    T = t1 - t0

    # Estimate ||A|| by probing
    probe_vecs = [y0.copy()]
    rng = np.random.default_rng(42)
    for _ in range(4):
        probe_vecs.append(rng.standard_normal(n))

    norm_est = 0.0
    for v in probe_vecs:
        vn = np.linalg.norm(v)
        if vn < 1e-15:
            continue
        Av = f(t0, v)
        ratio = np.linalg.norm(Av) / vn
        norm_est = max(norm_est, ratio)

    if norm_est < 1e-15:
        return y0.copy()

    # Find p such that |T| * norm_est / 2^p < 0.25
    p = 0
    while abs(T) * norm_est / (2 ** p) > 0.25:
        p += 1
    p = max(p, 0)

    T_scaled = T / (2 ** p)

    # Compute M = exp(T_scaled * A) by applying Taylor series to each basis vector
    # M[:, j] = exp(T_scaled * A) @ e_j = sum_{k=0}^{K} (T_scaled^k / k!) * A^k @ e_j

    max_terms = 50
    M = np.zeros((n, n))

    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        res_j = e_j.copy()
        term_j = e_j.copy()  # A^k @ e_j
        c_j = 1.0  # T_scaled^k / k!
        for k in range(1, max_terms):
            term_j = f(t0, term_j)  # A @ term_j = A^{k} @ e_j
            c_j *= T_scaled / k
            term_scaled_j = c_j * term_j
            res_j_new = res_j + term_scaled_j
            if np.linalg.norm(term_scaled_j) < 1e-16 * (1.0 + np.linalg.norm(res_j_new)):
                res_j = res_j_new
                break
            res_j = res_j_new
        M[:, j] = res_j

    # Apply M p times to y0
    y_current = y0.copy()
    for _ in range(p):
        y_current = M @ y_current

    return y_current
# EVOLVE-BLOCK-END
