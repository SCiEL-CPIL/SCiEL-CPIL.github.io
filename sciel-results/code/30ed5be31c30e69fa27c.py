def _compute_outer_side(inner_positions, inner_angles, outer_cx, outer_cy, outer_angle):
    """Compute minimum outer hexagon side needed to contain all inner hexagons."""
    max_side = 0.0
    for (cx, cy, ang) in zip(inner_positions, inner_angles):
        verts = _hex_vertices(cx, cy, 1.0, ang)
        # ...
