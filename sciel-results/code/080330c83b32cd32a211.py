<<<<<<< SEARCH
    # Run multiple restarts of hybrid search
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Generate initial points
        if restart == 0:
            # Simple random initialization
            points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        points[i] = [u + 0.5 * v, H * v]
                        break
        elif restart % 2 == 0 and best_points is not None:
            # Perturb the best found so far
            points = best_points + rng.gauss(0, 0.05, (N, 2))
            for i in range(N):
                points[i] = _project_to_triangle(points[i])
        else:
            # New random initialization
            points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        points[i] = [u + 0.5 * v, H * v]
                        break
        
        # 1. Short SA phase to get close to a local optimum
        sa_time_slice = min(2.0, deadline - 15.0 - time.time())
        if sa_time_slice <= 0: break
        sa_points, sa_score = _local_search_sa(points, time.time() + sa_time_slice)
        
        # 2. Smooth Polish phase
        polish_time_slice = min(3.0, deadline - 10.0 - time.time())
        if polish_time_slice <= 0: break
        pol_points, pol_score = _polish_smooth(sa_points, time.time() + polish_time_slice)
        
        # 3. Check if improvement
        if pol_score > best_score:
            best_points = pol_points
            best_score = pol_score
            
        # Also check SA result in case polish failed or didn't improve
        if sa_score > best_score:
            best_points = sa_points
            best_score = sa_score
=======
    # Run multiple restarts of hybrid search
    max_restarts = 25
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Generate initial points
        if restart == 0:
            # Structured initialization
            points = _structured_init()
        elif restart == 1:
            # Simple random initialization
            points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        points[i] = [u + 0.5 * v, H * v]
                        break
        elif restart % 2 == 0 and best_points is not None:
            # Perturb the best found so far
            points = best_points + rng.gauss(0, 0.05, (N, 2))
            points = _project_points(points)
        else:
            # New random initialization
            points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        points[i] = [u + 0.5 * v, H * v]
                        break
        
        # 1. Short SA phase to get close to a local optimum
        sa_time_slice = min(2.5, deadline - 20.0 - time.time())
        if sa_time_slice <= 0: break
        sa_points, sa_score = _local_search_sa(points, time.time() + sa_time_slice)
        
        # 2. Smooth Polish phase
        polish_time_slice = min(3.0, deadline - 15.0 - time.time())
        if polish_time_slice <= 0: break
        pol_points, pol_score = _polish_smooth(sa_points, time.time() + polish_time_slice)
        
        # 3. Coordinate descent polish
        cd_time_slice = min(2.0, deadline - 10.0 - time.time())
        if cd_time_slice <= 0: break
        cd_points, cd_score = _coordinate_descent_polish(pol_points, time.time() + cd_time_slice)
        
        # 4. Check if improvement
        if cd_score > best_score:
            best_points = cd_points
            best_score = cd_score
        elif pol_score > best_score:
            best_points = pol_points
            best_score = pol_score
        elif sa_score > best_score:
            best_points = sa_points
            best_score = sa_score
>>>>>>> REPLACE

