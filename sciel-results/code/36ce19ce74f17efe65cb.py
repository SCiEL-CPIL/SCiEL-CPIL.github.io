def _init_fcc():
    # 4x4x4 grid
    x = np.linspace(-0.45, 0.45, 4)
    y = np.linspace(-0.45, 0.45, 4)
    z = np.linspace(-0.45, 0.45, 4)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Select 50 points. 
    # Instead of random, let's pick the ones that form a dense cluster or just random.
    # Random is fine for diversity, but let's ensure they are distinct.
    idx = np.random.choice(len(grid), 50, replace=False)
    centers = grid[idx].copy()
    centers += np.random.normal(0, 0.01, size=centers.shape)
    return centers
