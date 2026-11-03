    def sat_overlap(v1, v2):
        """SAT check for two convex hexagons (each 6x2 array)."""
        for i in range(6):
            # Edge normal from v1
            edge = v1[i] - v1[(i+1) % 6]
            normal = np.array([-edge[1], edge[0]])
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return False
            # Edge normal from v2
            edge2 = v2[i] - v2[(i+1) % 6]
            normal2 = np.array([-edge2[1], edge2[0]])
            proj1b = v1 @ normal2
            proj2b = v2 @ normal2
            if proj1b.max() < proj2b.min() or proj2b.max() < proj1b.min():
                return False
        return True
