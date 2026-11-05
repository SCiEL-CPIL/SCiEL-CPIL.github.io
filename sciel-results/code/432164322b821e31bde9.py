    def make_cell_v3(param, shape_type, background):
        # background: 1 for dielectric bg, 0 for air bg
        cell = np.full((size, size), background, dtype=int)
        # feature_val is 1 if bg is 0, 0 if bg is 1
        feature_val = 1 - background
        
        if shape_type == 'circle':
            centers = [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    for cx, cy in centers:
                        if (x - cx)**2 + (y - cy)**2 < param**2:
                            cell[i, j] = feature_val
                            break
        elif shape_type == 'square':
            centers = [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    for cx, cy in centers:
                        if abs(x - cx) < param and abs(y - cy) < param:
                            cell[i, j] = feature_val
                            break
        return cell.tolist()
