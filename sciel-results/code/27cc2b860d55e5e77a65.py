# EVOLVE-BLOCK-START
import numpy as np
import time


def _project_entropy(P, r):
    """Compute Shannon entropy (in bits) of the projection x + r*y."""
    n = P.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    # P is indexed [y, x], so for cell (x, y) the value is P[y, x]
    # Projection value: x + r*y
    vals = x[None, :] + r * y[:, None]  # shape (10, 10), vals[y, x] = x + r*y
    # Pool by value
    unique_vals, inverse = np.unique(vals, return_inverse=True)
    probs = np.zeros(len(unique_vals))
    for i, p in enumerate(P.ravel()):
        if p > 0:
            probs[inverse.ravel()[i]] += p
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _grad_H_r(log_P, r, n=10):
    """
    Compute gradient of H_r with respect to log_P.
    H_r = -sum_v p_v log2(p_v) where p_v = sum_{(x,y): x+r*y=v} P[y,x]
    dH_r/dlog_P[y,x] = dH_r/dP[y,x] * P[y,x]
    dH_r/dP[y,x] = -log2(p_v) - 1/ln(2)  where v = x + r*y
    So dH_r/dlog_P[y,x] = P[y,x] * (-log2(p_v) - 1/ln(2))
    """
    P = np.exp(log_P)
    P = P / P.sum()  # normalize for safety
    x = np.arange(n)
    y = np.arange(n)
    vals = x[None, :] + r * y[:, None]  # vals[y, x] = x + r*y
    unique_vals, inverse = np.unique(vals, return_inverse=True)
    p_v = np.zeros(len(unique_vals))
    for i, p in enumerate(P.ravel()):
        p_v[inverse.ravel()[i]] += p
    # dH/dlog_P[y,x] = P[y,x] * (-log2(p_v) - 1/ln(2))
    log2_p_v = np.log2(np.maximum(p_v, 1e-300))
    grad_P = -log2_p_v[inverse] - 1.0 / np.log(2)
    grad_logP = P * grad_P
    return grad_logP


def _soft_max(Hs, temp):
    """Soft max of Hs with temperature temp."""
    if temp <= 0:
        return np.max(Hs)
    Hs = np.array(Hs)
    m = np.max(Hs)
    exps = np.exp((Hs - m) / temp)
    return m + temp * np.log(np.sum(exps))


def _soft_max_grad(Hs, temp):
    """Gradient of soft_max with respect to each H_i."""
    Hs = np.array(Hs)
    m = np.max(Hs)
    exps = np.exp((Hs - m) / temp)
    probs = exps / np.sum(exps)
    return probs


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximize D/M where D = H(x-y) and M = max(H(x), H(x+y), H(x+2y)).
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s limit

    n = 10
    # Parameterize P via log_P for unconstrained optimization
    # Initialize with uniform
    P_init = np.ones((n, n)) / (n * n)
    log_P = np.log(P_init)

    # Precompute projection indices
    x = np.arange(n)
    y = np.arange(n)
    slopes = [-1.0, 0.0, 1.0, 2.0]
    proj_indices = {}
    for r in slopes:
        vals = x[None, :] + r * y[:, None]
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        proj_indices[r] = (unique_vals, inverse)

    def compute_objective(log_P):
        P = np.exp(log_P)
        P = np.maximum(P, 0)
        s = P.sum()
        if s < 1e-300:
            return 0.0, P
        P = P / s
        H_minus1 = _project_entropy(P, -1.0)
        H_0 = _project_entropy(P, 0.0)
        H_1 = _project_entropy(P, 1.0)
        H_2 = _project_entropy(P, 2.0)
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0, P
        ratio = H_minus1 / M
        return ratio, P

    def compute_smoothed_objective(log_P, temp):
        P = np.exp(log_P)
        P = np.maximum(P, 0)
        s = P.sum()
        if s < 1e-300:
            return 0.0, P
        P = P / s
        H_minus1 = _project_entropy(P, -1.0)
        H_0 = _project_entropy(P, 0.0)
        H_1 = _project_entropy(P, 1.0)
        H_2 = _project_entropy(P, 2.0)
        Hs = np.array([H_0, H_1, H_2])
        M_smooth = _soft_max(Hs, temp)
        if M_smooth < 1e-9:
            return 0.0, P
        ratio = H_minus1 / M_smooth
        return ratio, P

    def compute_gradient(log_P, temp):
        """
        Compute gradient of smoothed ratio D/M_smooth with respect to log_P.
        """
        P = np.exp(log_P)
        P = np.maximum(P, 0)
        s = P.sum()
        P = P / s

        # Compute H values and their gradients
        H_minus1 = _project_entropy(P, -1.0)
        H_0 = _project_entropy(P, 0.0)
        H_1 = _project_entropy(P, 1.0)
        H_2 = _project_entropy(P, 2.0)
        Hs = np.array([H_0, H_1, H_2])
        M_smooth = _soft_max(Hs, temp)
        if M_smooth < 1e-9:
            return np.zeros_like(log_P)

        # Gradient of D = H_minus1
        grad_D = _grad_H_r(log_P, -1.0, n)

        # Gradient of M_smooth with respect to each H_i
        probs = _soft_max_grad(Hs, temp)
        # dM_smooth/dH_i = probs[i]
        # dM_smooth/dlog_P = sum_i probs[i] * dH_i/dlog_P
        grad_M = probs[0] * _grad_H_r(log_P, 0.0, n) + \
                 probs[1] * _grad_H_r(log_P, 1.0, n) + \
                 probs[2] * _grad_H_r(log_P, 2.0, n)

        # d(D/M)/dlog_P = (grad_D * M - D * grad_M) / M^2
        ratio = H_minus1 / M_smooth
        grad = (grad_D * M_smooth - H_minus1 * grad_M) / (M_smooth ** 2)

        return grad

    # Start with uniform
    best_ratio, best_P = compute_objective(log_P)

    # Gradient ascent with adaptive step size
    lr = 0.5
    temp = 0.1  # soft-max temperature
    patience = 0
    max_patience = 50
    min_lr = 0.001

    for it in range(5000):
        if time.time() > deadline:
            break

        # Compute gradient
        grad = compute_gradient(log_P, temp)

        # Normalize gradient for stable step
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break

        # Take step
        log_P_new = log_P + lr * grad

        # Evaluate
        ratio_new, P_new = compute_objective(log_P_new)

        if ratio_new > best_ratio + 1e-10:
            best_ratio = ratio_new
            best_P = P_new.copy()
            log_P = log_P_new
            patience = 0
            # Decay temperature to approach hard max
            temp = max(0.01, temp * 0.995)
        else:
            patience += 1
            lr *= 0.9
            if patience >= max_patience:
                break

        if lr < min_lr:
            break

    # Final polish: try a few different initializations and local refinements
    # Try concentrating on diagonal-like patterns
    if time.time() < deadline - 30:
        # Try several structured initializations
        structured_inits = []

        # Diagonal concentration: P[y,x] high when x - y is in a specific set
        for offset in range(-4, 5):
            P_struct = np.zeros((n, n))
            for yy in range(n):
                for xx in range(n):
                    if xx - yy == offset:
                        P_struct[yy, xx] = 1.0
            P_struct /= P_struct.sum()
            structured_inits.append(P_struct)

        # Anti-diagonal concentration
        for offset in range(0, 10):
            P_struct = np.zeros((n, n))
            for yy in range(n):
                for xx in range(n):
                    if xx + yy == offset:
                        P_struct[yy, xx] = 1.0
            P_struct /= P_struct.sum()
            structured_inits.append(P_struct)

        # Row concentration
        for row in range(n):
            P_struct = np.zeros((n, n))
            P_struct[row, :] = 1.0 / n
            structured_inits.append(P_struct)

        # Column concentration
        for col in range(n):
            P_struct = np.zeros((n, n))
            P_struct[:, col] = 1.0 / n
            structured_inits.append(P_struct)

        # Corner concentration
        for corner in [(0,0), (0,9), (9,0), (9,9)]:
            cy, cx = corner
            P_struct = np.zeros((n, n))
            for yy in range(n):
                for xx in range(n):
                    dist = abs(yy - cy) + abs(xx - cx)
                    P_struct[yy, xx] = np.exp(-dist / 2.0)
            P_struct /= P_struct.sum()
            structured_inits.append(P_struct)

        # Run short optimization from each structured init
        for P_init_struct in structured_inits:
            if time.time() > deadline:
                break
            log_P_s = np.log(np.maximum(P_init_struct, 1e-300))
            # Short gradient ascent
            lr_s = 0.3
            temp_s = 0.1
            for it_s in range(500):
                if time.time() > deadline:
                    break
                grad_s = compute_gradient(log_P_s, temp_s)
                grad_norm_s = np.linalg.norm(grad_s)
                if grad_norm_s < 1e-15:
                    break
                log_P_s_new = log_P_s + lr_s * grad_s
                ratio_s, P_s = compute_objective(log_P_s_new)
                if ratio_s > best_ratio + 1e-10:
                    best_ratio = ratio_s
                    best_P = P_s.copy()
                    log_P_s = log_P_s_new
                    temp_s = max(0.01, temp_s * 0.99)
                else:
                    lr_s *= 0.9
                    if lr_s < 0.001:
                        break

    # Ensure validity
    best_P = np.maximum(best_P, 0)
    total = best_P.sum()
    if total < 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total

    return best_P


# EVOLVE-BLOCK-END
