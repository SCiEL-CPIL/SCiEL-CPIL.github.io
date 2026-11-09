<<<<<<< SEARCH
    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = opt_params[3 * i], opt_params[3 * i + 1], opt_params[3 * i + 2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    normals, offsets = _outer_half_widths(outer_center, outer_side, outer_angle)
    cont_pen = _containment_penalty_vec(inner_verts_list, normals, offsets)
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                disj_pen += d

    if cont_pen > 1e-4 or disj_pen > 1e-4:
=======
    # Final validation (use hard max/min for validation, not soft)
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = opt_params[3 * i], opt_params[3 * i + 1], opt_params[3 * i + 2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    normals, offsets = _outer_half_widths(outer_center, outer_side, outer_angle)
    # Hard containment check
    cont_pen = 0.0
    for iv in inner_verts_list:
        M = iv @ normals.T - offsets
        viol = np.maximum(0.0, -M).max(axis=1)
        cont_pen += float(viol.sum())
    # Hard disjointness check
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # Exact SAT
            min_depth = float('inf')
            axes = []
            for poly in (inner_verts_list[i], inner_verts_list[j]):
                n = len(poly)
                for k in range(n):
                    p1 = poly[k]
                    p2 = poly[(k + 1) % n]
                    edge = p2 - p1
                    norm = np.linalg.norm(edge)
                    if norm < 1e-10:
                        continue
                    normal = np.array([-edge[1], edge[0]]) / norm
                    axes.append(normal)
            for axis in axes:
                proj1 = inner_verts_list[i] @ axis
                proj2 = inner_verts_list[j] @ axis
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 <= min2 or max2 <= min1:
                    min_depth = 0.0
                    break
                overlap = min(max1, max2) - max(min1, min2)
                if overlap < min_depth:
                    min_depth = overlap
            if min_depth > 0:
                disj_pen += min_depth

    if cont_pen > 1e-4 or disj_pen > 1e-4:
>>>>>>> REPLACE
