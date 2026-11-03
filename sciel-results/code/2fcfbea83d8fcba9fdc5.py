    def make_honeycomb(scale=1.0, offset_angle=0.0):
        # Scale lattice to ensure strict disjointness (small gap)
        gap_factor = 1.005 
        a1 = np.array([sqrt3, 0.0]) * scale * gap_factor
        a2 = np.array([sqrt3/2.0, 1.5]) * scale * gap_factor
    