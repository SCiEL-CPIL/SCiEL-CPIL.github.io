def _initialize_grid(W, H, kx, ky):
    cx = np.linspace(0, W, kx + 2)[1:-1]
    cy = np.linspace(0, H, ky + 2)[1:-1]
    
    grid_pts = []
    for y in cy:
        for x in cx:
            grid_pts.append([x, y])
            
    grid_pts = np.array(grid_pts)
    idx = np.random.permutation(len(grid_pts))
    selected = grid_pts[idx[:N]]
    
    # Spacing approx
    dx = W / (kx + 1)
    dy = H / (ky + 1)
    r_init = min(dx, dy) * 0.4
    r_init = max(r_init, 0.01)
    
    r = np.full(N, r_init)
    cx_out = selected[:, 0]
    cy_out = selected[:, 1]
    
    return np.column_stack([cx_out, cy_out, r])
