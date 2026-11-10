    # Final polish
    points = best_points.copy()
    r_current = best_r
    step_size = 0.01
    noise_temp = 0.01
    for _ in range(1000):
        if time.time() - start_time > timeout - 5:
            break
        forces, valid, _ = get_forces_and_validity_vectorized(points, r_current)
        if valid:
            r_current += 0.0001
        points += step_size * forces + np.random.normal(0, noise_temp, size=points.shape)
        _, new_valid, _ = get_forces_and_validity_vectorized(points, r_current)
        if new_valid and r_current > best_r:
            best_r = r_current
            best_points = points.copy()
            
    return best_points
