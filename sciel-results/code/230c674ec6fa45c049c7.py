<<<<<<< SEARCH
            penalty = 0.0
            verts = all_hex_vertices(inner_positions, inner_angles)
            for i in range(11):
                for j in range(i+1, 11):
                    dist = np.linalg.norm(inner_positions[i] - inner_positions[j])
                    if dist < 4.0:
                        gap = min_separation(verts[i], verts[j])
                        if gap < 1e-6:
                            penalty += 1000.0 * (1e-6 - gap) ** 2 + 100.0 * (1e-6 - gap)
                                
            if s < np.sqrt(11):
                s = np.sqrt(11)
            return s + penalty
=======
            penalty = 0.0
            verts = all_hex_vertices(inner_positions, inner_angles)
            for i in range(11):
                for j in range(i+1, 11):
                    dist = np.linalg.norm(inner_positions[i] - inner_positions[j])
                    if dist < 4.0:
                        gap = min_separation(verts[i], verts[j])
                        if gap < 1e-6:
                            penalty += 10000.0 * (1e-6 - gap)
                                
            if s < np.sqrt(11):
                s = np.sqrt(11)
            return s + penalty
>>>>>>> REPLACE
