from scipy.optimize import basinhopping

def basin_hopping_step(bary_init, time_limit=5.0):
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    def step(x):
        # Random step within bounds
        step_size = 0.05
        x_new = x + np.random.normal(0, step_size, size=x.shape)
        # Clamp to bounds roughly, clamp_bary will fix it
        return clamp_bary(x_new.reshape(N, 2)).flatten()
        
    # Note: basinhopping minimizes a function. We want to maximize min area, so minimize -min area (or soft min).
    # Use soft_min_obj for the main function to be smooth.
    
    try:
        res = basinhopping(
            soft_min_obj,
            bary_init.flatten(),
            stepfun=step,
            minimizer_kwargs={'bounds': bounds, 'method': 'L-BFGS-B', 'options': {'maxiter': 100}},
            niter=10,
            T=100.0,
            accept_test=None,
            disp=False
        )
        if res.fun < soft_min_obj(bary_init.flatten(), 50.0):
            return clamp_bary(res.x.reshape(N, 2))
    except Exception:
        pass
    return bary_init
