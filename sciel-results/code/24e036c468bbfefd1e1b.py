<<<<<<< SEARCH
def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    ge = _get_ground_energy(n, coupling, field)
    H = _get_hamiltonian(n, coupling, field)
    
    # If we have very little time, return the seed solution
    if time.time() > deadline - 10:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    theta = _optimize_one(n, coupling, field, layers, ge, H, deadline)
    
    # Ensure finite values
    theta = np.where(np.isfinite(theta), theta, 0.0)
    
    return theta
=======
def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    ge = _get_ground_energy(n, coupling, field)
    H = _get_hamiltonian(n, coupling, field)
    
    # If we have very little time, return the seed solution
    if time.time() > deadline - 15:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    theta = _optimize_one(n, coupling, field, layers, ge, H, deadline)
    
    # Ensure finite values
    theta = np.where(np.isfinite(theta), theta, 0.0)
    
    return theta
>>>>>>> REPLACE
