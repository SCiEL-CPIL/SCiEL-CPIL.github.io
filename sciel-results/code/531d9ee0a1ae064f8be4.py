def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_radius = np.pi
    
    # Generate a few initial configurations
    inits = []
    inits.append(_fibonacci_spiral(N))
    
    # Add a random configuration
    rng = np.random.default_rng(42)
    rand_pts = rng.standard_normal((N, 3))
    inits.append(_normalize(rand_pts))
    
    for init in inits:
        points = _normalize(init)
        
        # Run Lloyd's algorithm
        # Use a smaller step size for stability
        step_size = 0.1
        
        while time.time() < deadline:
            # Run a batch of iterations
            for _ in range(100):
                points, success = _lloyd_iteration(points, step_size)
                if not success:
                    break
            
            if not success:
                break
                
            # Optional: Check for convergence
            # If the max movement is very small, break
            
        r = _covering_radius(points)
        if r < best_radius:
            best_radius = r
            best_points = points
            
        if time.time() > deadline:
            break
            
    if best_points is None:
        return _fibonacci_spiral(N)
        
    return best_points
