        max_proj = 0.0
        for k in range(6):
            angle = theta_rad + math.pi/6 + k * math.pi/3
            nx = math.cos(angle)
            ny = math.sin(angle)
            for (x, y) in points:
                proj = x * nx + y * ny
                if abs(proj) > abs(max_proj):
                    max_proj = proj
        
        return 2.0 * abs(max_proj) / sqrt3
