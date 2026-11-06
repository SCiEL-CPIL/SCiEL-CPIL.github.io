<<<<<<< SEARCH
def _precompute_references(models, a_values, num_steps=5000):
    """
    Pre-compute true f(a) and D(a) for all models and a values using RK4.
    """
    refs = []
    for (Om0, w0, wa) in models:
        a_min = np.min(a_values)
        x_grid = np.linspace(0, np.log(a_min), num_steps + 1)
        a_grid = np.exp(x_grid)
        dx = x_grid[1] - x_grid[0]
        
        f_vals = np.zeros_like(x_grid)
        ln_D_vals = np.zeros_like(x_grid)
        f_vals[0] = 0.5  # f(1) for matter-only at a=1
        ln_D_vals[0] = 0.0
        
        for i in range(num_steps):
            a_curr = a_grid[i]
            Om_m_curr = _Om_m_of_a(a_curr, Om0, w0, wa)
            f_curr = f_vals[i]
            
            # RK4 for df/dx = -f(2+f) + 1.5*Om_m
            def dfdx(f):
                return -f * (2.0 + f) + 1.5 * Om_m_curr
            
            k1 = dfdx(f_curr)
            k2 = dfdx(f_curr + 0.5 * dx * k1)
            k3 = dfdx(f_curr + 0.5 * dx * k2)
            k4 = dfdx(f_curr + dx * k3)
            f_vals[i+1] = f_curr + (dx / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Integrate ln_D: d(lnD)/dx = f
            ln_D_vals[i+1] = ln_D_vals[i] + 0.5 * (f_curr + f_vals[i+1]) * dx
        
        # Interpolate to a_values
        x_targets = np.log(a_values)
        f_true = np.interp(x_targets, x_grid, f_vals)
        ln_D_true = np.interp(x_targets, x_grid, ln_D_vals)
        D_true = np.exp(ln_D_true)
        
        refs.append({
            'f_true': f_true,
            'D_true': D_true,
            'Om_m': _Om_m_of_a(a_values, Om0, w0, wa),
            'w0': w0,
            'wa': wa
        })
    return refs
=======
def _precompute_references(models, a_values, num_steps=5000):
    """
    Pre-compute true f(a) and D(a) for all models and a values using RK4.
    """
    refs = []
    for (Om0, w0, wa) in models:
        a_min = np.min(a_values)
        x_grid = np.linspace(0, np.log(a_min), num_steps + 1)
        a_grid = np.exp(x_grid)
        dx = x_grid[1] - x_grid[0]
        
        f_vals = np.zeros_like(x_grid)
        ln_D_vals = np.zeros_like(x_grid)
        f_vals[0] = 0.5  # f(1) for matter-only at a=1
        ln_D_vals[0] = 0.0
        
        for i in range(num_steps):
            a_curr = a_grid[i]
            Om_m_curr = _Om_m_of_a(a_curr, Om0, w0, wa)
            f_curr = f_vals[i]
            
            # RK4 for df/dx = -f(2+f) + 1.5*Om_m
            def dfdx(f, Om_m_val=Om_m_curr):
                return -f * (2.0 + f) + 1.5 * Om_m_val
            
            k1 = dfdx(f_curr)
            k2 = dfdx(f_curr + 0.5 * dx * k1)
            k3 = dfdx(f_curr + 0.5 * dx * k2)
            k4 = dfdx(f_curr + dx * k3)
            f_vals[i+1] = f_curr + (dx / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Integrate ln_D: d(lnD)/dx = f
            ln_D_vals[i+1] = ln_D_vals[i] + 0.5 * (f_curr + f_vals[i+1]) * dx
        
        # Interpolate to a_values
        x_targets = np.log(a_values)
        f_true = np.interp(x_targets, x_grid, f_vals)
        ln_D_true = np.interp(x_targets, x_grid, ln_D_vals)
        D_true = np.exp(ln_D_true)
        
        refs.append({
            'f_true': f_true,
            'D_true': D_true,
            'Om_m': _Om_m_of_a(a_values, Om0, w0, wa),
            'w0': w0,
            'wa': wa,
            'Om0': Om0
        })
    return refs
>>>>>>> REPLACE

Now let me fix the `_local_error_fast` to include a proper D approximation:

