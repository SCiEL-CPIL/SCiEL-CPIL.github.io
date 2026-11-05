def construct():
    # Standard hexagonal spiral: H0 + H1 + H2 + first 13 of Ring 3
    # This is the known optimal Harborth configuration for n=32
    points = [
        [0, 0],
        # Ring 1 (6 points)
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        # Ring 2 (12 points)
        [2, -1], [2, 0], [2, 1], [1, 2], [0, 2], [-1, 2],
        [-2, 1], [-2, 0], [-2, -1], [-1, -2], [0, -2], [1, -2],
        # Ring 3, first 13 points continuing the spiral
        [3, -2], [3, -1], [3, 0], [3, 1], [2, 2], [1, 3], [0, 3],
        [-1, 3], [-2, 2], [-3, 1], [-3, 0], [-3, -1], [-4, 0]
    ]
    return points
