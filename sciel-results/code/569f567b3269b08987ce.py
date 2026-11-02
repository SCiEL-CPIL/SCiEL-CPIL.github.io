<<<<<<< SEARCH
def _initial_config_rings():
    """Concentric rings initial configuration."""
    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.22 * np.cos(angle)
        y = 0.5 + 0.22 * np.sin(angle)
        positions.append([x, y, 0.10 * 0.9])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.35 * np.cos(angle)
        y = 0.5 + 0.35 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.48 * np.cos(angle)
        y = 0.5 + 0.48 * np.sin(angle)
        positions.append([x, y, 0.06 * 0.9])
    return np.array(positions[:N])
=======
def _initial_config_rings():
    """Concentric rings initial configuration."""
    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.22 * np.cos(angle)
        y = 0.5 + 0.22 * np.sin(angle)
        positions.append([x, y, 0.10 * 0.9])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.35 * np.cos(angle)
        y = 0.5 + 0.35 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.48 * np.cos(angle)
        y = 0.5 + 0.48 * np.sin(angle)
        positions.append([x, y, 0.06 * 0.9])
    return np.array(positions[:N])

def _initial_config_corner_biased():
    """Initial configuration biased toward corners and edges."""
    positions = []
    # Corner regions
    corner_positions = [
        (0.12, 0.12, 0.10), (0.88, 0.12, 0.10),
        (0.12, 0.88, 0.10), (0.88, 0.88, 0.10),
    ]
    for x, y, r in corner_positions:
        positions.append([x, y, r])
    
    # Edge midpoints
    edge_positions = [
        (0.5, 0.10, 0.09), (0.5, 0.90, 0.09),
        (0.10, 0.5, 0.09), (0.90, 0.5, 0.09),
    ]
    for x, y, r in edge_positions:
        positions.append([x, y, r])
    
    # Center region
    center_positions = [
        (0.5, 0.5, 0.12),
        (0.35, 0.35, 0.08), (0.65, 0.35, 0.08),
        (0.35, 0.65, 0.08), (0.65, 0.65, 0.08),
    ]
    for x, y, r in center_positions:
        positions.append([x, y, r])
    
    # Fill remaining with smaller circles in gaps
    gap_positions = [
        (0.25, 0.25, 0.07), (0.75, 0.25, 0.07),
        (0.25, 0.75, 0.07), (0.75, 0.75, 0.07),
        (0.5, 0.25, 0.06), (0.5, 0.75, 0.06),
        (0.25, 0.5, 0.06), (0.75, 0.5, 0.06),
    ]
    for x, y, r in gap_positions:
        positions.append([x, y, r])
    
    return np.array(positions[:N])

def _initial_config_diagonal():
    """Diagonal band initial configuration."""
    positions = []
    # Main diagonal
    for k in range(9):
        t = (k + 0.5) / 9
        x = 0.1 + 0.8 * t
        y = 0.1 + 0.8 * t
        r = 0.08
        positions.append([x, y, r])
    
    # Anti-diagonal
    for k in range(9):
        t = (k + 0.5) / 9
        x = 0.1 + 0.8 * t
        y = 0.9 - 0.8 * t
        r = 0.08
        positions.append([x, y, r])
    
    # Center fill
    positions.append([0.5, 0.5, 0.10])
    
    return np.array(positions[:N])
>>>>>>> REPLACE

