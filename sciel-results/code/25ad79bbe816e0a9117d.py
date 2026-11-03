<<<<<<< SEARCH
def _objective(x, w, h, lambda_rad=0.5):
    """
    Objective function for optimization.
    x = [x1, y1, r1, x2, y2, r2, ...]
    We minimize: (Sum of Squared Violations) - lambda_rad * (Sum of Radii)
    This encourages large radii, penalized by overlap/wall breaches.
    """
    n = N
    pos = x[:n*2].reshape(n, 2)
    radii = x[n*2:]
    
    # Compute violations
    wall_viol = 0.0
    for i in range(n):
        wall_viol += max(0, radii[i] - pos[i, 0]) ** 2
        wall_viol += max(0, radii[i] - (w - pos[i, 0])) ** 2
        wall_viol += max(0, radii[i] - pos[i, 1]) ** 2
        wall_viol += max(0, radii[i] - (h - pos[i, 1])) ** 2
        
    circle_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d_sq = dx*dx + dy*dy
            # Optimization: check d_sq > (r_i+r_j)^2 first to avoid sqrt
            r_sum = radii[i] + radii[j]
            if d_sq < r_sum * r_sum:
                d = np.sqrt(d_sq)
                overlap = r_sum - d
                circle_viol += overlap * overlap
                
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    
    return total_viol - lambda_rad * sum_rad
=======
def _penalty_objective(x, w, h, penalty_weight=100.0):
    """
    Penalty-based objective for simulated annealing.
    x = [x1, y1, r1, x2, y2, r2, ...]
    Returns total penalty (lower is better).
    """
    n = N
    pos = x[:n*2].reshape(n, 2)
    radii = x[n*2:]
    
    # Compute violations
    wall_viol = 0.0
    for i in range(n):
        wall_viol += max(0.0, radii[i] - pos[i, 0]) ** 2
        wall_viol += max(0.0, radii[i] - (w - pos[i, 0])) ** 2
        wall_viol += max(0.0, radii[i] - pos[i, 1]) ** 2
        wall_viol += max(0.0, radii[i] - (h - pos[i, 1])) ** 2
        
    circle_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d_sq = dx*dx + dy*dy
            r_sum = radii[i] + radii[j]
            if d_sq < r_sum * r_sum:
                d = np.sqrt(d_sq)
                overlap = r_sum - d
                circle_viol += overlap * overlap
                
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    
    # Penalty objective: minimize violations, maximize radii
    return penalty_weight * total_viol - sum_rad
>>>>>>> REPLACE

