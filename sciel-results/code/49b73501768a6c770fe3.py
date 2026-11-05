<<<<<<< SEARCH
def _optimize_thicknesses(materials, x0, problem, wls, n_mat, deadline, max_iter=200):
    """
    Nelder-Mead optimization for thicknesses of a fixed material sequence.
    """
    n = len(x0)
    lo = np.zeros(n)
    hi = np.full(n, problem["max_layer_nm"])
    max_total = problem["max_total_nm"]
    
    # Ensure initial guess is valid
    x = np.array(x0, dtype=float)
    x = np.clip(x, lo, hi)
    total = np.sum(x)
    if total > max_total:
        x = x * (max_total / total)
        x = np.clip(x, lo, hi)
        
    def objective(x):
        # Check total thickness constraint strictly during optimization
        if np.sum(x) > max_total:
            # Penalize proportionally to help the optimizer back off
            return 1e6 + (np.sum(x) - max_total) * 10.0
        return _mean_reflectance_percent(materials, list(x), problem, wls, n_mat)

    # Initialize simplex
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        # Step size: 10% of range or 50nm, whichever is smaller
        step = min(50.0, 0.1 * (hi[i] - lo[i]))
        if step < 1.0: step = 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
        
    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for iter_count in range(max_iter):
        if time.time() > deadline:
            break
            
        # Sort simplex
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Check convergence
        spread = abs(f_vals[0] - f_vals[-1])
        if spread < 1e-6:
            break
        
        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
            
        # Expansion
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
            
        # Contraction
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue
            
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]
=======
def _reflectance_and_grad(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array and gradient with respect to thicknesses.
    Uses the characteristic matrix method.
    Returns (R_array, grad) where grad has shape (n_layers, n_wls).
    """
    n_layers = len(materials)
    n_wls = len(wls)
    
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        R = np.full(n_wls, r * r)
        return R, np.zeros((0, n_wls))
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d != 0:
            beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)
    
    # Store intermediate matrices for gradient computation
    # M_prefix[i] = M_1 * M_2 * ... * M_i
    # M_suffix[i] = M_{i+1} * ... * M_N
    M_prefix = [None] * n_layers
    M_suffix = [None] * n_layers
    
    # Compute prefix matrices
    cM11 = np.ones(n_wls, dtype=complex)
    cM12 = np.zeros(n_wls, dtype=complex)
    cM21 = np.zeros(n_wls, dtype=complex)
    cM22 = np.ones(n_wls, dtype=complex)
    
    for i in range(n_layers):
        m = materials[i]
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c
        
        nM11 = cM11 * A + cM12 * C
        nM12 = cM11 * B + cM12 * D
        nM21 = cM21 * A + cM22 * C
        nM22 = cM21 * B + cM22 * D
        
        cM11, cM12, cM21, cM22 = nM11, nM12, nM21, nM22
        M_prefix[i] = (cM11.copy(), cM12.copy(), cM21.copy(), cM22.copy())
    
    M11, M12, M21, M22 = cM11, cM12, cM21, cM22
    
    # Compute suffix matrices (from right to left)
    cM11 = np.ones(n_wls, dtype=complex)
    cM12 = np.zeros(n_wls, dtype=complex)
    cM21 = np.zeros(n_wls, dtype=complex)
    cM22 = np.ones(n_wls, dtype=complex)
    
    for i in range(n_layers - 1, -1, -1):
        m = materials[i]
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c
        
        # M_suffix[i] = M_i * M_suffix[i+1]
        # But we want M_suffix[i] such that M_total = M_prefix[i-1] * M_i * M_suffix[i]
        # Actually, let's define M_suffix[i] = M_{i+1} * ... * M_N
        # So M_total = M_prefix[i] * M_suffix[i] where M_prefix[i] includes M_i
        
        # Let me redefine:
        # M_prefix[i] = M_1 * ... * M_i
        # M_suffix[i] = M_{i+1} * ... * M_N
        # M_total = M_prefix[i] * M_suffix[i]
        
        # To compute M_suffix[i], we multiply M_i * M_suffix[i+1]
        # But M_suffix[i] should be M_{i+1} * ... * M_N
        
        # Let's use a different approach:
        # M_suffix[i] = M_{i+1} * ... * M_N
        # M_suffix[n_layers-1] = I
        # M_suffix[i] = M_{i+1} * M_suffix[i+1]
        
        if i == n_layers - 1:
            M_suffix[i] = (np.ones(n_wls, dtype=complex), np.zeros(n_wls, dtype=complex),
                          np.zeros(n_wls, dtype=complex), np.ones(n_wls, dtype=complex))
        else:
            next_suf = M_suffix[i+1]
            ns11, ns12, ns21, ns22 = next_suf
            
            # M_i * M_suffix[i+1]
            sM11 = A * ns11 + B * ns21
            sM12 = A * ns12 + B * ns22
            sM21 = C * ns11 + D * ns21
            sM22 = C * ns12 + D * ns22
            
            M_suffix[i] = (sM11, sM12, sM21, sM22)
        
        # Update cumulative suffix for next iteration
        # cM represents M_{i+1} * ... * M_N
        # We need to update it to M_i * M_{i+1} * ... * M_N
        # But actually, we just need M_suffix[i] for each i
        
    # Now compute reflectance
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    # Compute gradient
    # dR/dt_i = 2 * Re(conj(r) * dr/dt_i)
    # dr/dt_i = d/dt_i [(n_ambient - Y) / (n_ambient + Y)]
    # = [-(n_ambient + Y) * dY/dt_i - (n_ambient - Y) * dY/dt_i] / (n_ambient + Y)^2
    # = -2 * dY/dt_i / (n_ambient + Y)^2
    
    # dY/dt_i where Y = (M21 + n_sub * M22) / (M11 + n_sub * M12)
    # = [(dM21/dt + n_sub * dM22/dt)(M11 + n_sub * M12) - (M21 + n_sub * M22)(dM11/dt + n_sub * dM12/dt)] / (M11 + n_sub * M12)^2
    
    # dM_j/dt_i where M = M_prefix[i-1] * M_i * M_suffix[i]
    # = M_prefix[i-1] * dM_i/dt * M_suffix[i]
    
    # dM_i/dt_i: M_i = [[cos(beta_i), i*sin(beta_i)/n_i], [i*n_i*sin(beta_i), cos(beta_i)]]
    # beta_i = 2*pi*n_i*t_i/wl
    # dbeta/dt = 2*pi*n_i/wl
    # dM_i/dt = [[-sin(beta_i)*dbeta/dt, i*cos(beta_i)*dbeta/dt/n_i],
    #            [i*n_i*cos(beta_i)*dbeta/dt, -sin(beta_i)*dbeta/dt]]
    
    grad = np.zeros((n_layers, n_wls))
    
    for i in range(n_layers):
        m = materials[i]
        n = n_mat[m]
        dbeta_dt = 2.0 * math.pi * n / wls  # shape (n_wls,)
        
        c = cosb[i]
        s = sinb[i]
        
        # dM_i/dt_i
        dA = -s * dbeta_dt
        dB = 1j * c * dbeta_dt / n
        dC = 1j * n * c * dbeta_dt
        dD = -s * dbeta_dt
        
        # Get prefix and suffix
        if i == 0:
            p11 = np.ones(n_wls, dtype=complex)
            p12 = np.zeros(n_wls, dtype=complex)
            p21 = np.zeros(n_wls, dtype=complex)
            p22 = np.ones(n_wls, dtype=complex)
        else:
            p11, p12, p21, p22 = M_prefix[i-1]
        
        s11, s12, s21, s22 = M_suffix[i]
        
        # dM/dt = P * dM_i * S
        # P = [[p11, p12], [p21, p22]]
        # dM_i = [[dA, dB], [dC, dD]]
        # S = [[s11, s12], [s21, s22]]
        
        # First compute dM_i * S
        dMS11 = dA * s11 + dB * s21
        dMS12 = dA * s12 + dB * s22
        dMS21 = dC * s11 + dD * s21
        dMS22 = dC * s12 + dD * s22
        
        # Then P * (dM_i * S)
        dM11 = p11 * dMS11 + p12 * dMS21
        dM12 = p11 * dMS12 + p12 * dMS22
        dM21 = p21 * dMS11 + p22 * dMS21
        dM22 = p21 * dMS12 + p22 * dMS22
        
        # Now compute dY/dt
        # Y = (M21 + n_sub * M22) / (M11 + n_sub * M12)
        # dY/dt = [(dM21 + n_sub * dM22)(M11 + n_sub * M12) - (M21 + n_sub * M22)(dM11 + n_sub * dM12)] / (M11 + n_sub * M12)^2
        
        num = (dM21 + n_substrate * dM22) * (M11 + n_substrate * M12) - \
              (M21 + n_substrate * M22) * (dM11 + n_substrate * dM12)
        den = (M11 + n_substrate * M12) ** 2
        dY_dt = num / den
        
        # dr/dt = -2 * dY_dt / (n_ambient + Y)^2
        dr_dt = -2.0 * dY_dt / (n_ambient + Y) ** 2
        
        # dR/dt = 2 * Re(conj(r) * dr_dt)
        grad[i, :] = 2.0 * np.real(np.conj(r) * dr_dt)
    
    return R, grad

def _optimize_l_bfgs(materials, x0, problem, wls, n_mat, deadline, max_iter=100):
    """
    L-BFGS optimization with gradient for thicknesses of a fixed material sequence.
    """
    n = len(x0)
    lo = np.zeros(n)
    hi = np.full(n, problem["max_layer_nm"])
    max_total = problem["max_total_nm"]
    
    x = np.array(x0, dtype=float)
    x = np.clip(x, lo, hi)
    total = np.sum(x)
    if total > max_total:
        x = x * (max_total / total)
        x = np.clip(x, lo, hi)
    
    def objective_and_grad(x):
        if np.sum(x) > max_total:
            # Penalize and provide gradient for penalty
            penalty = 1e6 + (np.sum(x) - max_total) * 10.0
            # Gradient of penalty is 10 for each component
            R, _ = _reflectance_and_grad(materials, list(x), problem["ambient_index"],
                                         problem["substrate_index"], wls, n_mat)
            f = float(np.mean(R)) * 100.0 + penalty
            grad = np.full(n, 10.0)
            return f, grad
        
        R, grad_arr = _reflectance_and_grad(materials, list(x), problem["ambient_index"],
                                            problem["substrate_index"], wls, n_mat)
        f = float(np.mean(R)) * 100.0
        grad = np.mean(grad_arr, axis=1) * 100.0
        return f, grad
    
    # Simple L-BFGS implementation
    m = 5  # Memory size
    history_x = []
    history_f = []
    history_g = []
    
    f, g = objective_and_grad(x)
    best_x = x.copy()
    best_f = f
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        if np.linalg.norm(g) < 1e-6:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        if len(history_x) == 0:
            d = -g
        else:
            # Two-loop recursion
            q = g.copy()
            alphas = []
            rhos = []
            
            # First loop (backwards)
            for i in range(len(history_x) - 1, -1, -1):
                s = history_x[i] - history_x[i-1] if i > 0 else history_x[i] - x
                y = history_g[i] - (history_g[i-1] if i > 0 else g)
                if np.dot(y, s) < 1e-10:
                    continue
                rho = 1.0 / np.dot(y, s)
                alpha = rho * np.dot(s, q)
                q = q - alpha * y
                alphas.append(alpha)
                rhos.append(rho)
            
            # Initial Hessian approximation
            if len(rhos) > 0:
                yy = np.dot(history_g[-1], history_g[-1])
                ss = np.dot(history_x[-1] - (history_x[-2] if len(history_x) > 1 else x), 
                           history_x[-1] - (history_x[-2] if len(history_x) > 1 else x))
                gamma = ss / yy if yy > 1e-10 else 1.0
            else:
                gamma = 1.0
            
            r = gamma * q
            
            # Second loop (forwards)
            for i in range(len(alphas) - 1, -1, -1):
                s = history_x[i] - history_x[i-1] if i > 0 else history_x[i] - x
                y = history_g[i] - (history_g[i-1] if i > 0 else g)
                beta = rhos[i] * np.dot(y, r)
                r = r + (alphas[i] - beta) * s
            
            d = -r
        
        # Line search (backtracking)
        step = 1.0
        for ls in range(20):
            x_new = x + step * d
            x_new = np.clip(x_new, lo, hi)
            f_new, _ = objective_and_grad(x_new)
            
            if f_new < f:
                break
            step *= 0.5
        
        if f_new < best_f:
            best_x = x_new.copy()
            best_f = f_new
        
        # Update history
        history_x.append(x)
        history_f.append(f)
        history_g.append(g)
        
        if len(history_x) > m:
            history_x.pop(0)
            history_f.pop(0)
            history_g.pop(0)
        
        x = x_new
        f, g = objective_and_grad(x)
    
    return np.clip(best_x, lo, hi), best_f

def _optimize_thicknesses(materials, x0, problem, wls, n_mat, deadline, max_iter=200):
    """
    Optimized thickness optimization using L-BFGS with gradient.
    """
    # Try L-BFGS first
    x_opt, f_opt = _optimize_l_bfgs(materials, x0, problem, wls, n_mat, deadline, max_iter=100)
    
    # Post-process: Ensure constraints
    x_opt = np.clip(x_opt, 0.0, problem["max_layer_nm"])
    total_opt = sum(x_opt)
    if total_opt > problem["max_total_nm"]:
        factor = problem["max_total_nm"] / total_opt
        x_opt = x_opt * factor
        x_opt = np.clip(x_opt, 0.0, problem["max_layer_nm"])
    
    # Calculate final score
    val = _mean_reflectance_percent(materials, list(x_opt), problem, wls, n_mat)
    
    return x_opt, val
>>>>>>> REPLACE
