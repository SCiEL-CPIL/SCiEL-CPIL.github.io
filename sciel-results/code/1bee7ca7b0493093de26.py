    def has_q3_in_8set(verts, u=None, v=None):
        if u is not None:
            idx_map = {vert: i for i, vert in enumerate(verts)}
            idx_u = idx_map.get(u)
            idx_v = idx_map.get(v)
            if idx_u is None or idx_v is None:
                return None # Should not happen
        for A_idx, B_idx in partitions:
            if u is not None:
                if (idx_u in A_idx) == (idx_v in A_idx):
                    continue
            # ...
