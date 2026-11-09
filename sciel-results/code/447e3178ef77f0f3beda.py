def objective(params):
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:]
    
    # Boundary penalties
    pen_x_min = np.maximum(0, r - x)**2
    pen_x_max = np.maximum(0, x - (1 - r))**2
    pen_y_min = np.maximum(0, r - y)**2
    pen_y_max = np.maximum(0, y - (1 - r))**2
    boundary_pen = np.sum(pen_x_min + pen_x_max + pen_y_min + pen_y_max) * 100
    
    # Overlap penalties
    # Compute distance matrix
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_d = r[:, None] + r[None, :]
    # We only care about upper triangle
    iu = np.triu_indices(N, 1)
    overlaps = np.maximum(0, min_d[iu] - dist[iu])**2
    overlap_pen = np.sum(overlaps) * 100
    
    # Radius positivity
    pen_r = np.sum(np.maximum(0, -r)**2) * 100
    
    return -np.sum(r) + boundary_pen + overlap_pen + pen_r
