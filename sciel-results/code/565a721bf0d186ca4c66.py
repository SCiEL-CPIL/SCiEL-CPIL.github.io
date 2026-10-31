<<<<<<< SEARCH
def relax(positions, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_func, x0, method='L-BFGS-B', 
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
        return result.x.reshape(N, 3)
    except Exception:
        return positions
=======
def relax(positions, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_func, x0, method='L-BFGS-B', 
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def relax_multi_start(positions, n_starts=3, noise_scale=0.02, maxiter=200):
    """Try multiple small perturbations and keep the best relaxed result."""
    best_pos = positions
    best_energy = lj_energy_vectorized(positions)
    
    rng = np.random.default_rng(42)
    
    for i in range(n_starts):
        if time.time() - START_TIME > DEADLINE - 20:
            break
        noise = rng.normal(0, noise_scale, positions.shape)
        perturbed = positions + noise
        if not check_validity(perturbed):
            continue
        relaxed = relax(perturbed, maxiter=maxiter, gtol=1e-7)
        if check_validity(relaxed):
            e = lj_energy_vectorized(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed
    return best_pos, best_energy
>>>>>>> REPLACE

