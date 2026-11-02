            # Calculate centroid of inner hexagon centers
            if len(inner_list) > 0:
                cx_mean = sum(p[0] for p in inner_list) / len(inner_list)
                cy_mean = sum(p[1] for p in inner_list) / len(inner_list)
                center_inner = np.array([cx_mean, cy_mean])
            else:
                center_inner = np.array([0.0, 0.0])
            
            centers = [
                verts.mean(axis=0),
                center_inner,
                np.array([0.0, 0.0])
            ]
