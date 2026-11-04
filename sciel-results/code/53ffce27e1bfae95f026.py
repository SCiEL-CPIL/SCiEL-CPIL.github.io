    ca = np.cos(alpha + np.arange(6) * math.pi / 3.0)
    sa = np.sin(alpha + np.arange(6) * math.pi / 3.0)
    # normals is (2, 6) to allow (66, 2) @ (2, 6) -> (66, 6)
    normals = np.stack([ca, sa], axis=0)
    dots = verts @ normals
    extents = dots.max(axis=0) - dots.min(axis=0)
    return extents.max() / d
