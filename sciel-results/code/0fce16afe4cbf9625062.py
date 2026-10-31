    while time.time() - start_time < time_budget:
        # Generate perturbation
        noise = rng.normal(0, step_size, size=current_bary.shape)
        new_bary = current_bary + noise
