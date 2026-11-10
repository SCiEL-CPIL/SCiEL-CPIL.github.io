<<<<<<< SEARCH
    # Build domain and values from non-zero mu values
    mask = mu != 0
    domain = np.where(mask)[0]
    domain = domain[domain > 0]  # ensure positive
    values = mu[domain].astype(np.float64)
    
    return domain, values
=======
    # Build domain and values from non-zero mu values
    # We use f(k) = -mu(k) because sum mu(k) log(k)/k -> 1, so a = -sum -> -1.
    # Flipping sign makes a -> 1.
    mask = mu != 0
    domain = np.where(mask)[0]
    domain = domain[domain > 0]  # ensure positive
    values = -mu[domain].astype(np.float64)
    
    return domain, values
>>>>>>> REPLACE
