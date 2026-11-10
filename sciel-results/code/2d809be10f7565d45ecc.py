def _check_inside_hexagon(inner_verts, outer_verts, tol=1e-6):
    """Check if all vertices of inner hexagons are inside or on the boundary of the outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        ex = edge[0]
        ey = edge[1]
        
        # Vectorized cross product check for all 66 vertices
        # inner_verts shape: (11, 6, 2)
        # p1 shape: (2,)
        # diff shape: (11, 6, 2)
        diff = inner_verts - p1
        # cross z-component: ex * dy - ey * dx
        cross_vals = ex * diff[:, :, 1] - ey * diff[:, :, 0]
        
        if np.any(cross_vals < -tol):
            return False
    return True
