    def make_cluster():
        pts = [[0, 0]]
        # 6 neighbors
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)])
        # 5 more neighbors from the next ring
        # The next ring has 12 points.
        # We want to pick 5.
        # To be symmetric, we can't.
        # Let's pick 5 that are "closest" to the axis of symmetry?
        # Or just pick the first 5.
        # Actually, a 3x4 rectangle is 12 points.
        # A 2x6 rectangle is 12 points.
        # Let's trust the rectangles.
        return np.array(pts[:12])
