<<<<<<< SEARCH
def _optimize_packing(centers, R, time_budget):
    """
    Optimize centers for a fixed radius R using Langevin dynamics.
    """
    start_time = time.time()
    current_centers = centers.copy()
    velocities = np.zeros_like(current_centers)
    
    # Tuned for speed: larger dt, fewer steps
    dt = 0.05
    temp = 0.05
    damping = 0.5
    max_steps = 200
    
    # Simulate
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
            
        current_centers, velocities = _langevin_step(current_centers, velocities, R, dt, temp, damping)
        
        # Cool down temperature over time
        progress = step / max_steps
        temp = 0.05 * (1.0 - progress) + 0.001
        # Increase damping as we cool to settle
        damping = 0.5 + 0.5 * progress
    
    # Final gradient descent polish
    def obj(x):
        e, _ = _energy_and_grad(x, R)
        return e
        
    def jac(x):
        _, g = _energy_and_grad(x, R)
        return g
        
    try:
        res = minimize(obj, current_centers.flatten(), jac=jac, method='L-BFGS-B', 
                       options={'maxiter': 100, 'disp': False})
        current_centers = res.x.reshape(-1, 3)
        current_centers = np.clip(current_centers, -0.499999, 0.499999)
    except Exception:
        pass
        
    return current_centers
=======
def _polish_centers(centers, R, time_budget, beta=100.0):
    """
    High-precision polish pass using L-BFGS-B with tight tolerances.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        e, g = _energy_and_grad(x, R, beta)
        return e, g
    
    try:
        res = minimize(obj_and_grad, centers.flatten(), jac=True, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False,
                                'maxls': 50, 'maxfun': 500})
        polished = res.x.reshape(-1, 3)
        polished = np.clip(polished, -0.4999999, 0.4999999)
    except Exception:
        polished = centers
    
    return polished

def _optimize_packing(centers, R, time_budget):
    """
    Optimize centers for a fixed radius R using Langevin dynamics + polish.
    """
    start_time = time.time()
    current_centers = centers.copy()
    velocities = np.zeros_like(current_centers)
    
    # Tuned for speed
    dt = 0.03
    temp = 0.03
    damping = 0.8
    max_steps = 300
    beta = 50.0
    
    # Simulate
    for step in range(max_steps):
        if time.time() - start_time > time_budget * 0.6:
            break
            
        current_centers, velocities = _langevin_step(current_centers, velocities, R, dt, temp, damping, beta)
        
        # Cool down temperature over time
        progress = step / max_steps
        temp = 0.03 * (1.0 - progress) + 0.0005
        # Increase damping as we cool to settle
        damping = 0.8 + 0.2 * progress
    
    # Final gradient descent polish with more time
    time_left = time_budget - (time.time() - start_time)
    if time_left > 0.5:
        current_centers = _polish_centers(current_centers, R, time_left, beta=100.0)
    
    return current_centers
>>>>>>> REPLACE

