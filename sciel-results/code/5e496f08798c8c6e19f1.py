<<<<<<< SEARCH
    # Strategy: Run multiple SA restarts with different initial configurations
    sa_time_budget = DEADLINE * 0.65
    
    # Multiple initial configurations for restarts
    def make_honeycomb(scale=1.0, offset_angle=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N) + offset_angle
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles
    
    def make_ring_config():
        # 12 hexagons in a ring pattern
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            theta = 2 * np.pi * i / N
            r = 2.5
            positions[i, 0] = r * np.cos(theta)
            positions[i, 1] = r * np.sin(theta)
            angles[i] = theta * 180 / np.pi
        return positions, angles
    
    def make_two_rows():
        # Two rows of 6
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(6):
            positions[i, 0] = i * sqrt3
            positions[i, 1] = 0
            positions[i+6, 0] = (i + 0.5) * sqrt3
            positions[i+6, 1] = 1.5
        return positions, angles
    
    initial_configs = [
        make_honeycomb(1.0, 0.0),
        make_honeycomb(1.0, 15.0),
        make_honeycomb(0.9, 0.0),
        make_ring_config(),
        make_two_rows(),
    ]
    
    # Run SA with multiple restarts
    num_restarts = 3
    restart_time = sa_time_budget / num_restarts
=======
    # Strategy: Run multiple SA restarts with different initial configurations
    sa_time_budget = DEADLINE * 0.60
    
    # Multiple initial configurations for restarts
    def make_honeycomb(scale=1.0, offset_angle=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N) + offset_angle
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles
    
    def make_ring_config():
        # 12 hexagons in a ring pattern
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            theta = 2 * np.pi * i / N
            r = 2.5
            positions[i, 0] = r * np.cos(theta)
            positions[i, 1] = r * np.sin(theta)
            angles[i] = theta * 180 / np.pi
        return positions, angles
    
    def make_two_rows():
        # Two rows of 6
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(6):
            positions[i, 0] = i * sqrt3
            positions[i, 1] = 0
            positions[i+6, 0] = (i + 0.5) * sqrt3
            positions[i+6, 1] = 1.5
        return positions, angles
    
    def make_compact_cluster():
        # Compact cluster: 3-4-3-2 arrangement
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        s3 = sqrt3
        # Row 0: 3 hexagons
        for i in range(3):
            positions[i, 0] = i * s3
            positions[i, 1] = 0
        # Row 1: 4 hexagons
        for i in range(4):
            positions[3+i, 0] = (i + 0.5) * s3 - s3/2
            positions[3+i, 1] = 1.5
        # Row 2: 3 hexagons
        for i in range(3):
            positions[7+i, 0] = i * s3
            positions[7+i, 1] = 3.0
        # Row 3: 2 hexagons
        for i in range(2):
            positions[10+i, 0] = (i + 0.5) * s3
            positions[10+i, 1] = 4.5
        return positions, angles
    
    initial_configs = [
        make_honeycomb(1.0, 0.0),
        make_honeycomb(1.0, 15.0),
        make_honeycomb(0.9, 0.0),
        make_ring_config(),
        make_two_rows(),
        make_compact_cluster(),
        make_honeycomb(0.95, 30.0),
        make_honeycomb(1.05, 0.0),
    ]
    
    # Run SA with multiple restarts
    num_restarts = 4
    restart_time = sa_time_budget / num_restarts
>>>>>>> REPLACE

