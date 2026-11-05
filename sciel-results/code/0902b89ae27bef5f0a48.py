                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                verts = [u, v1, v2, v3]
                min_v = min(verts)
                # ...
                e1 = edge_map[(min(u, v1), max(u, v1))]
                e2 = edge_map[(min(u, v2), max(u, v2))]
                e3 = edge_map[(min(v1, v3), max(v1, v3))]
                e4 = edge_map[(min(v2, v3), max(v2, v3))]
