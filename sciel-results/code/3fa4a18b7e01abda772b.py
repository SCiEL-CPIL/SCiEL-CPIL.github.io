    def objective(x):
        points = x.reshape((N, 2))
        for p in points:
            if not is_inside_triangle(p):
                return 1e10 # Large positive penalty
        return -min_triangle_area(points)
