    def construct():
        # Optimal solution: Place 1 unit hexagon in the center.
        # Minimal outer side length is 1.
        inner = [[0.0, 0.0, 0.0]]
        outer_center = (0.0, 0.0)
        side = 1.0
        angle = 0.0
        return inner, outer_center, side, angle
    