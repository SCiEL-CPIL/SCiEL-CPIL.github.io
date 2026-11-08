    def _min_side(center, angle_deg):
        verts = _hex_verts(center, angle_deg) # (6,2)
        return np.max(verts @ NORMALS)  # (6,2) @ (6,) -> ERROR
    