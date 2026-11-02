<<<<<<< SEARCH
    def initial_guess():
        """Generate a valid initial honeycomb packing."""
        params = np.zeros(3*N + 3)
        
        # Honeycomb grid positions for 12 hexagons
        # Use axial coordinates for hex grid
        sqrt3 = np.sqrt(3)
        
        # Center at origin, then 6 neighbors, then 5 more in second ring
        positions = []
        # Center
        positions.append((0, 0, 0.0))
        # First ring: 6 hexagons
        # Distance between centers of touching unit hexagons is 2 * apothem = sqrt(3) * 2 ?
        # Apothem = sqrt(3)/2. Distance = 2 * sqrt(3)/2 = sqrt(3).
        # Wait, side length is 1. Apothem is sqrt(3)/2.
        # Center-to-center distance for touching is 2 * apothem = sqrt(3).
        # Let's verify.
        # Hexagon side 1. Width (flat-to-flat) is sqrt(3).
        # So distance is sqrt(3).
        for i in range(6):
            theta = i * np.pi / 3
            r = np.sqrt(3)
            positions.append((r * np.cos(theta), r * np.sin(theta), 0.0))
        # Second ring: pick 5 positions
        # Distance for second ring neighbors: 2 * sqrt(3)
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            r = 2 * np.sqrt(3)
            positions.append((r * np.cos(theta), r * np.sin(theta), 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params
=======
    def initial_guess():
        """Generate a valid initial honeycomb packing using axial coordinates."""
        params = np.zeros(3*N + 3)
        sqrt3 = np.sqrt(3)
        
        # Use axial coordinates for hex grid: (q, r) -> (x, y)
        # x = sqrt(3) * (q + r/2), y = 1.5 * r
        # For unit hexagons, center-to-center distance when touching is sqrt(3)
        
        # Axial coordinates for a compact 12-hexagon cluster
        # Center + 6 neighbors + 5 more from second ring
        axial_coords = [
            (0, 0),      # center
            (1, 0),      # 6 neighbors
            (0, 1),
            (-1, 1),
            (-1, 0),
            (0, -1),
            (1, -1),
            (2, 0),      # 5 from second ring
            (1, 1),
            (-1, 2),
            (-2, 1),
            (-2, 0),
        ]
        
        positions = []
        for q, r in axial_coords:
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            positions.append((x, y, 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params
>>>>>>> REPLACE

