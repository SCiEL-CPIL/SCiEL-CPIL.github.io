    # Inside the loop
    if i > 0 and i % 3 == 0:
        # Perturb best
        c_pert = best_x + np.random.normal(0, 0.05, (N, 2))
        c_pert = np.clip(c_pert, 0.01, 0.99)
        r_pert = np.clip(best_r * 0.9, 0.01, 0.4)
    elif i % 2 == 0:
        # Perturb greedy
        ...
    else:
        # Random
        ...
