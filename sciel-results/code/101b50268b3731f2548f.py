    stop_flag = {'stop': False}
    def callback(xk):
        if time.time() > deadline:
            stop_flag['stop'] = True
        return stop_flag['stop']

    # ...
    result = opt.minimize(
        energy_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        callback=callback,
        options={'maxiter': 1000, 'gtol': 1e-8, 'maxls': 20}
    )
