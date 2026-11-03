<<<<<<< SEARCH
    # Honeycomb initial configuration
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0
=======
    # Honeycomb initial configuration - tighter packing
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0
    
    # Try to improve initial config slightly with a quick local optimization
    def quick_local_improve(pos, ang, time_limit=1.0):
        t_start = time.time()
        current_side, _, _, _ = compute_min_outer_side(pos, ang)
        best_side = current_side
        best_pos = pos.copy()
        best_ang = ang.copy()
        step = 0.1
        while time.time() - t_start < time_limit and step > 0.005:
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_pos.copy()
                        new_ang = best_ang.copy()
                        if dim == 0:
                            new_pos[idx, 0] += direction * step
                        elif dim == 1:
                            new_pos[idx, 1] += direction * step
                        else:
                            new_ang[idx] += direction * step
                        angles_rad_new = np.radians(new_ang)
                        h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                        valid = True
                        for j in range(N):
                            if j == idx:
                                continue
                            h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                            if not hexagons_disjoint_fast(h_moved, h_other):
                                valid = False
                                break
                        if not valid:
                            continue
                        new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side - 1e-10:
                            best_pos = new_pos
                            best_ang = new_ang
                            best_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.7
        return best_side, best_pos, best_ang
    
    init_side, best_positions, best_angles = quick_local_improve(base_positions, base_angles, 2.0)
>>>>>>> REPLACE

