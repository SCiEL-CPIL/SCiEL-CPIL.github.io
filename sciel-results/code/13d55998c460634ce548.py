<<<<<<< SEARCH
    def generate_rectangular_grid():
        """Generate a 3x4 rectangular grid of hexagons."""
        dx = math.sqrt(3) + 0.01
        dy = 3.0 + 0.01
        centers = []
        angles = []
        for row in range(3):
            for col in range(4):
                cx = (col - 1.5) * dx
                cy = (row - 1.0) * dy
                centers.append([cx, cy])
                angles.append(30.0)
        return centers, angles
=======
    def generate_rectangular_grid():
        """Generate a 3x4 rectangular grid of hexagons."""
        dx = math.sqrt(3) + 0.05
        dy = 3.0 + 0.05
        centers = []
        angles = []
        for row in range(3):
            for col in range(4):
                cx = (col - 1.5) * dx
                cy = (row - 1.0) * dy
                centers.append([cx, cy])
                angles.append(0.0)
        return centers, angles
>>>>>>> REPLACE

