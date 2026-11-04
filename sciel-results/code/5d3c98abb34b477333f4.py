    def outer_side(data, ocx, ocy, otheta_deg):
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            na = otheta + math.pi/6.0 + k * math.pi/3.0
            nx = math.cos(na); ny = math.sin(na)
            for i in range(N):
                verts = data[i][0]
                for (vx, vy) in verts:
                    proj = (vx-ocx)*nx + (vy-ocy)*ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / SQRT3
