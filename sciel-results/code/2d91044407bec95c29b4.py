<<<<<<< SEARCH
def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary
=======
def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def generate_grid_init():
    """
    Generate a grid-based initialization for diversity.
    Places points on a regular grid inside the triangle.
    """
    best_bary = None
    best_val = -np.inf
    
    for n_grid in [3, 4, 5]:
        # Create grid in barycentric coordinates
        u_vals = np.linspace(0.1, 0.8, n_grid)
        v_vals = np.linspace(0.1, 0.8, n_grid)
        
        points = []
        for u in u_vals:
            for v in v_vals:
                if u + v <= 0.9 and len(points) < N:
                    points.append([u, v])
        
        # If we don't have enough, fill randomly
        if len(points) < N:
            rng = np.random.default_rng(42 + n_grid)
            while len(points) < N:
                u = rng.random() * 0.8 + 0.1
                v = rng.random() * 0.8 + 0.1
                if u + v <= 0.9:
                    points.append([u, v])
        
        bary = np.array(points[:N])
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
    
    if best_bary is None:
        return generate_hexagonal_init()
    
    return best_bary
>>>>>>> REPLACE

