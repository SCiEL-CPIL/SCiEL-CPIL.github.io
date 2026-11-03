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
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        pts = np.array(pts)
        # Scale slightly from centroid to ensure strict disjointness (avoid touching)
        centroid = np.mean(pts, axis=0)
        pts = centroid + (pts - centroid) * 1.0001
        return pts
>>>>>>> REPLACE
