<<<<<<< SEARCH
    # Compute (z_grid - mu_i) / sigma_i for all i
    # This is (2000, n_quad)
    z_diff = (z_grid[None, :] - mu[:, None]) / sigma_safe[:, None]
    
    # Standard normal pdf
    pdf_vals = np.exp(-0.5 * z_diff ** 2) / np.sqrt(2 * np.pi)
    
    # Divide by sigma
    integrand = pdf_vals / sigma_safe[:, None]
    
    # Multiply by 1/z_grid
    integrand = integrand * (1.0 / z_grid[None, :])
    
    # Integrate using trapezoidal rule
    # integral_i = sum_j integrand[i,j] * dz[j]
    integral = np.sum(integrand * dz[None, :], axis=1)
=======
    # Compute (z_grid - mu_i) / sigma_i for all i
    # This is (2000, n_quad)
    z_diff = (z_grid[None, :] - mu[:, None]) / sigma_safe[:, None]
    
    # Standard normal pdf
    pdf_vals = np.exp(-0.5 * z_diff ** 2) / np.sqrt(2 * np.pi)
    
    # Divide by sigma
    integrand = pdf_vals / sigma_safe[:, None]
    
    # Multiply by 1/z_grid
    integrand = integrand * (1.0 / z_grid[None, :])
    
    # Integrate using trapezoidal rule
    # integral_i = sum_j integrand[i,j] * dz[j]
    # Use np.trapz for proper trapezoidal integration
    integral = np.trapz(integrand, z_grid[None, :], axis=1)
>>>>>>> REPLACE
