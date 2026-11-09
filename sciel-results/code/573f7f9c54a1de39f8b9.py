<<<<<<< SEARCH
    # Initial 3x4 grid
    init_points = [[x, y] for x in range(4) for y in range(3)]
    
    best_points = [p[:] for p in init_points]
    best_score = count_ordinary_lines(best_points)
    
    random.seed(42)
=======
    # Initial Hexagonal lattice subset
    # Center + Ring 1 + 5 points from Ring 2
    hex_points = [
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [0, 2], [-2, 2], [-2, 0], [0, -2]
    ]
    init_points = hex_points
    
    best_points = [p[:] for p in init_points]
    best_score = count_ordinary_lines(best_points)
    
    random.seed(42)
>>>>>>> REPLACE
