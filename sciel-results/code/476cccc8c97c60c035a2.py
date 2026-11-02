        for x, y, angle in inner:
            rad = np.deg2rad(angle)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            # Base vertices at angle 0
            base_verts = [
                (1.0, 0.0),
                (0.5, sqrt3 / 2.0),
                (-0.5, sqrt3 / 2.0),
                (-1.0, 0.0),
                (-0.5, -sqrt3 / 2.0),
                (0.5, -sqrt3 / 2.0)
            ]
            verts = []
            for vx0, vy0 in base_verts:
                # Rotate
                vx = x + vx0 * cos_a - vy0 * sin_a
                vy = y + vx0 * sin_a + vy0 * cos_a
                verts.append((vx, vy))
  