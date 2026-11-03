<<<<<<< SEARCH
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
=======
    def axial_to_cartesian(axial_list):
        pts = []
        # Scale factor to ensure strict disjointness (gap between hexagons)
        # If "disjoint" implies no touching, a small epsilon is needed.
        # If touching is allowed, this has negligible cost.
        scale = 1.001 
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0) * scale
            y = 1.5 * r * scale
            pts.append((x, y))
        return np.array(pts)
>>>>>>> REPLACE
