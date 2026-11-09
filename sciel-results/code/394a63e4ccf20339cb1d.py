<<<<<<< SEARCH
def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell maximizing the TM band gap
    between bands 4 and 5.
    """
    t_start = time.time()
    DEADLINE = 225.0  # Safety margin before 230s limit
    
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")
    
    # Set up PW solver
    n_pw = 17  # 17x17 = 289 plane waves
    k_points = None  # Will be generated inside solver
    
    compute_bands = _build_pw_solver(size, eps_low, eps_high, n_pw, k_points)
    
    # Generate initial cells
    initial_cells = _generate_initial_cells(size)
    
    # Evaluate all initial cells
    best_score = -np.inf
    best_cell = None
    for cell in initial_cells:
        if time.time() - t_start > DEADLINE - 10:
            break
        score = compute_bands(cell.astype(np.float64))
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Local search from best initial cell
    remaining_time = DEADLINE - (time.time() - t_start) - 5.0
    if remaining_time > 5.0 and best_cell is not None:
        best_cell, best_score = _local_search(
            best_cell, size, compute_bands, time_limit=remaining_time * 0.4, n_iter=5000
        )
    
    # If we have more time, try local search from other good initial cells
    remaining_time = DEADLINE - (time.time() - t_start) - 5.0
    if remaining_time > 10.0:
        for cell in initial_cells:
            if time.time() - t_start > DEADLINE - 5:
                break
            if np.array_equal(cell, best_cell):
                continue
            t_sub = time.time()
            cell2, score2 = _local_search(
                cell, size, compute_bands, time_limit=min(remaining_time / 3, 25.0), n_iter=2000
            )
            if score2 > best_score:
                best_score = score2
                best_cell = cell2
    
    # Final hill climb from best cell
    remaining_time = DEADLINE - (time.time() - t_start) - 3.0
    if remaining_time > 5.0 and best_cell is not None:
        best_cell, best_score = _hill_climb(
            best_cell, size, compute_bands, time_limit=remaining_time, n_iter=10000
        )
    
    # Final check
    if best_cell is None:
        # Fallback: return a simple rod
        best_cell = np.zeros((size, size), dtype=int)
        center = size / 2.0
        radius = size * 0.25
        for i in range(size):
            for j in range(size):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                if dx*dx + dy*dy <= radius*radius:
                    best_cell[i, j] = 1
        best_cell = _apply_symmetry(best_cell, size)
    
    return best_cell.tolist()
=======
def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell maximizing the TM band gap
    between bands 4 and 5.
    """
    t_start = time.time()
    DEADLINE = 225.0  # Safety margin before 230s limit
    
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")
    
    # Set up PW solver
    n_pw = 17  # 17x17 = 289 plane waves
    k_points = None  # Will be generated inside solver (reduced to 3 points for speed)
    
    compute_bands = _build_pw_solver(size, eps_low, eps_high, n_pw, k_points)
    
    # Generate initial cells
    initial_cells = _generate_initial_cells(size)
    
    # Evaluate all initial cells
    best_score = -np.inf
    best_cell = None
    for cell in initial_cells:
        if time.time() - t_start > DEADLINE - 10:
            break
        score = compute_bands(cell.astype(np.float64))
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Local search from best initial cell
    # Reduced iterations due to time constraints, but speed is improved by fewer k-points
    remaining_time = DEADLINE - (time.time() - t_start) - 5.0
    if remaining_time > 5.0 and best_cell is not None:
        # Allocate 60% of remaining time to first search
        best_cell, best_score = _local_search(
            best_cell, size, compute_bands, time_limit=remaining_time * 0.6, n_iter=2000
        )
    
    # If we have more time, try local search from other good initial cells
    remaining_time = DEADLINE - (time.time() - t_start) - 5.0
    if remaining_time > 10.0:
        # Try a few other initial cells
        for cell in initial_cells[:3]:
            if time.time() - t_start > DEADLINE - 5:
                break
            if np.array_equal(cell, best_cell):
                continue
            cell2, score2 = _local_search(
                cell, size, compute_bands, time_limit=min(remaining_time / 3, 15.0), n_iter=1000
            )
            if score2 > best_score:
                best_score = score2
                best_cell = cell2
    
    # Final hill climb from best cell
    remaining_time = DEADLINE - (time.time() - t_start) - 3.0
    if remaining_time > 5.0 and best_cell is not None:
        best_cell, best_score = _hill_climb(
            best_cell, size, compute_bands, time_limit=remaining_time, n_iter=5000
        )
    
    # Final check
    if best_cell is None:
        # Fallback: return a simple rod
        best_cell = np.zeros((size, size), dtype=int)
        center = size / 2.0
        radius = size * 0.25
        for i in range(size):
            for j in range(size):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                if dx*dx + dy*dy <= radius*radius:
                    best_cell[i, j] = 1
        best_cell = _apply_symmetry(best_cell, size)
    
    return best_cell.tolist()
>>>>>>> REPLACE
