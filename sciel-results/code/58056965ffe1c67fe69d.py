<<<<<<< SEARCH
    # Tight L-BFGS
    if time.time() < remaining_deadline - 5.0:
        best_pts_global = _lbfgs_tight_polish(best_pts_global, remaining_deadline - 5.0, maxiter=800)
        best_energy_global = _energy_only(best_pts_global)
        
    # One more standard L-BFGS with different settings?
    if time.time() < remaining_deadline - 1.0:
        best_pts_global = _lbfgs_polish(best_pts_global, remaining_deadline - 1.0, maxiter=200)
        best_energy_global = _energy_only(best_pts_global)
=======
    # Tight L-BFGS
    if time.time() < remaining_deadline - 8.0:
        best_pts_global = _lbfgs_tight_polish(best_pts_global, remaining_deadline - 8.0, maxiter=800)
        best_energy_global = _energy_only(best_pts_global)
        
    # Local perturbation restart for final escape from local minima
    if time.time() < remaining_deadline - 4.0:
        best_pts_global, best_energy_global = _local_perturb_restart(
            best_pts_global, best_energy_global, remaining_deadline - 4.0,
            n_perturb=5, perturb_scale=0.03)
        
    # Gradient descent polish for fine-tuning
    if time.time() < remaining_deadline - 1.0:
        best_pts_global, best_energy_global = _gradient_descent_polish(
            best_pts_global, remaining_deadline - 1.0, max_iter=300,
            lr_start=0.0005, lr_end=0.00002)
        
    # Final tight L-BFGS
    if time.time() < remaining_deadline - 0.5:
        best_pts_global = _lbfgs_tight_polish(best_pts_global, remaining_deadline - 0.5, maxiter=300)
        best_energy_global = _energy_only(best_pts_global)
>>>>>>> REPLACE
