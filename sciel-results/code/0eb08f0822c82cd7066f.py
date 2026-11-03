<<<<<<< SEARCH
def _grid_initialization(cols, rows, w, h, n_circles):
    """Create grid-based initial positions and radii."""
    xs = np.linspace(w / (2*cols), w - w / (2*cols), cols) if cols > 1 else np.array([w/2])
    ys = np.linspace(h / (2*rows), h - h / (2*rows), rows) if rows > 1 else np.array([h/2])
    
    positions = []
    for j in range(rows):
        for i in range(cols):
            positions.append([xs[i], ys[j]])
    
    positions = positions[:n_circles]
    positions = np.array(positions, dtype=float)
    
    dx = w / cols if cols > 1 else w
    dy = h / rows if rows > 1 else h
    base_r = min(dx, dy) * 0.45
    
    radii = np.ones(n_circles) * base_r
    
    return positions, radii
=======
def _hexagonal_initialization(w, h, n_circles, seed=0):
    """
    Create hexagonal lattice initial positions and radii.
    This provides a high-quality near-optimal starting configuration.
    """
    np.random.seed(seed)
    
    # Try to fit hexagonal packing
    # Hexagonal: rows offset, spacing = 2r horizontally, sqrt(3)*r vertically
    best_pos = None
    best_r = 0.0
    
    # Try various base radii
    for base_r in np.linspace(0.02, min(w, h) * 0.3, 30):
        spacing_x = 2.0 * base_r
        spacing_y = np.sqrt(3) * base_r
        
        positions = []
        row = 0
        while len(positions) < n_circles:
            offset = (row % 2) * spacing_x / 2.0
            col = 0
            while len(positions) < n_circles:
                x = spacing_x / 2.0 + offset + col * spacing_x
                y = spacing_y / 2.0 + row * spacing_y
                if x + base_r <= w and y + base_r <= h and x - base_r >= 0 and y - base_r >= 0:
                    positions.append([x, y])
                col += 1
            row += 1
            if row > 50:
                break
        
        if len(positions) >= n_circles:
            positions = np.array(positions[:n_circles])
            radii = np.ones(n_circles) * base_r
            if _is_valid(positions, radii, w, h, tol=1e-6):
                if base_r > best_r:
                    best_r = base_r
                    best_pos = positions
    
    if best_pos is None:
        # Fallback to grid
        return _grid_initialization(5, 5, w, h, n_circles)
    
    return best_pos, np.ones(n_circles) * best_r


def _grid_initialization(cols, rows, w, h, n_circles):
    """Create grid-based initial positions and radii."""
    xs = np.linspace(w / (2*cols), w - w / (2*cols), cols) if cols > 1 else np.array([w/2])
    ys = np.linspace(h / (2*rows), h - h / (2*rows), rows) if rows > 1 else np.array([h/2])
    
    positions = []
    for j in range(rows):
        for i in range(cols):
            positions.append([xs[i], ys[j]])
    
    positions = positions[:n_circles]
    positions = np.array(positions, dtype=float)
    
    dx = w / cols if cols > 1 else w
    dy = h / rows if rows > 1 else h
    base_r = min(dx, dy) * 0.45
    
    radii = np.ones(n_circles) * base_r
    
    return positions, radii
>>>>>>> REPLACE

