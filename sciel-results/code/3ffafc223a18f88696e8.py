        for r in range(3):
            for i in range(4):
                if r % 2 == 0:
                    centers.append((i * d_x, r * d_y))
                else:
                    centers.append((i * d_x + d_x/2, r * d_y))
