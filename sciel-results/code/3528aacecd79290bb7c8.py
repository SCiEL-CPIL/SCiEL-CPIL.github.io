    def _hexagons_overlap(v1, v2):
        # Check vertices
        for p in v1:
            if _point_in_hexagon(p[0], p[1], v2[:, 0].mean(), v2[:, 1].mean(), 1.0, 0.0): # Wrong center/angle!
                 pass
        # We need the center and angle of the hexagons to use _point_in_hexagon.
        # v1 and v2 are vertices. We can recover center and angle?
        # Center is mean of vertices.
        # Angle is harder.
        # Better: Pass center and angle to the check function, or store them.
    