<<<<<<< SEARCH
def _soft_min_area_cart(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, we can use max subtraction, but areas are small.
    # areas are in range [0, 0.433]. beta is large.
    # max_a = np.max(areas)
    # shifted = -beta * (areas - max_a)
    # log_sum = np.log(np.sum(np.exp(shifted)))
    # soft_min = max_a - log_sum / beta
    
    # Simpler and more stable for small areas:
    # Since areas > 0, we can just compute directly if beta is not too huge.
    # But let's stick to the stable version.
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip shifted to avoid underflow issues in exp if beta is very large
    # exp(-large) -> 0. This is fine.
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min
=======
def _soft_min_area_cart(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _soft_min_area_and_grad_cart(pts, beta):
    """
    Soft minimum of triangle areas AND its gradient w.r.t. pts (shape (N,2)).
    Returns (soft_min_value, grad) where grad[i] = d(soft_min)/d(pts[i]).
    
    soft_min = max_a - (1/beta)*log(sum(exp(-beta*areas)))
    d(soft_min)/d(area_k) = -(1/beta) * (-beta * exp(-beta*area_k)) / sum(exp(-beta*areas))
                           = exp(-beta*area_k) / sum(exp(-beta*areas))
                           = softmax_weight_k (over the -beta*areas)
    
    So d(soft_min)/d(pts[i]) = sum_k [weight_k * d(area_k)/d(pts[i])]
    
    For area_k = 0.5 * |cross_k| where cross_k = (Pj-Pi) x (Pk-Pi):
    d(area_k)/d(Pi) = -0.5 * sign(cross_k) * (Pk - Pj) rotated 90deg
    d(area_k)/d(Pj) =  0.5 * sign(cross_k) * (Pk - Pi) rotated 90deg  
    d(area_k)/d(Pk) = -0.5 * sign(cross_k) * (Pj - Pi) rotated 90deg
    
    Actually let me be more careful. area = 0.5*|cross|, cross = dx1*dy2 - dy1*dx2
    where dx1 = xj-xi, dy1 = yj-yi, dx2 = xk-xi, dy2 = yk-yi.
    
    d(cross)/d(xi) = -(dy2 - dy1) ... let me just compute numerically-free:
    cross = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    d(cross)/d(xi) = -(yk-yi) + (yj-yi) = yi - yk + yj - yi = yj - yk
    d(cross)/d(xj) = (yk-yi)
    d(cross)/d(xk) = -(yj-yi) = yi - yj
    d(cross)/d(yi) = -(xj-xi) + (xk-xi) = xi - xj + xk - xi = xk - xj
    d(cross)/d(yj) = -(xk-xi) = xi - xk
    d(cross)/d(yk) = (xj-xi)
    
    For area = 0.5*|cross|, d(area)/d = 0.5*sign(cross)*d(cross)/d
    
    So d(soft_min)/d(pts[i]) = sum_k weight_k * 0.5 * sign(cross_k) * d(cross_k)/d(pts[i])
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # these are the softmax weights for the soft-min
    
    log_sum = np.log(sum_exp)
    soft_min = max_a - log_sum / beta
    
    # Gradient: d(soft_min)/d(pts[i]) = sum_k weights[k] * d(area_k)/d(pts[i])
    # d(area_k)/d(pts[i]) = 0.5 * sign(cross_k) * d(cross_k)/d(pts[i])
    sign_cross = np.sign(cross)
    half_sign = 0.5 * sign_cross
    
    # d(cross)/d(xi) = yj - yk, d(cross)/d(yi) = xk - xj
    dci_x = half_sign * (pj[:, 1] - pk[:, 1])  # d(area)/d(xi)
    dci_y = half_sign * (pk[:, 0] - pj[:, 0])  # d(area)/d(yi)
    
    # d(cross)/d(xj) = yk - yi, d(cross)/d(yj) = xi - xk
    dcj_x = half_sign * (pk[:, 1] - pi[:, 1])  # d(area)/d(xj)
    dcj_y = half_sign * (pi[:, 0] - pk[:, 0])  # d(area)/d(yj)
    
    # d(cross)/d(xk) = yi - yj, d(cross)/d(yk) = xj - xi
    dck_x = half_sign * (pi[:, 1] - pj[:, 1])  # d(area)/d(xk)
    dck_y = half_sign * (pj[:, 0] - pi[:, 0])  # d(area)/d(yk)
    
    grad = np.zeros((N, 2))
    # Accumulate contributions
    np.add.at(grad, (_T_I, 0), weights * dci_x)
    np.add.at(grad, (_T_I, 1), weights * dci_y)
    np.add.at(grad, (_T_J, 0), weights * dcj_x)
    np.add.at(grad, (_T_J, 1), weights * dcj_y)
    np.add.at(grad, (_T_K, 0), weights * dck_x)
    np.add.at(grad, (_T_K, 1), weights * dck_y)
    
    return soft_min, grad
>>>>>>> REPLACE

