import numpy as np

N = 11

def construct():
    # Honeycomb lattice vectors for flat-top hexagons of side 1
    sqrt3 = np.sqrt(3)
    v1 = np.array([1.5, sqrt3 / 2])
    v2 = np.array([1.5, -sqrt3 / 2])

    # Generate candidate lattice points
    points = []
    max_coord = 20
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            pos = i * v1 + j * v2
            dist = np.linalg.norm(pos)
            points.append((dist, pos))

    # Sort by distance from origin and take the N closest
    points.sort(key=lambda x: x[0])
    selected = points[:N]

    # Unit flat-top hexagon vertices relative to center
    hex_vertices_rel = []
    for k in range(6):
        angle = np.radians(60 * k)
        hex_vertices_rel.append((np.cos(angle), np.sin(angle)))

    # For each selected hexagon, compute the max S needed for all vertices
    max_S = 0.0
    for (dist, pos) in selected:
        for (vx_rel, vy_rel) in hex_vertices_rel:
            vx = pos[0] + vx_rel
            vy = pos[1] + vy_rel
            # Minimal side length S of a flat-top regular hexagon centered at origin
            # that contains the point (vx, vy)
            s_req = max(
                abs(vx),
                abs(vy) / sqrt3,
                abs(vx + vy / sqrt3),
                abs(-vx + vy / sqrt3)
            )
            if s_req > max_S:
                max_S = s_req

    # Build inner list: (x, y, angle_degrees)
    inner = []
    for (dist, pos) in selected:
        inner.append([float(pos[0]), float(pos[1]), 0.0])

    return inner, (0.0, 0.0), float(max_S), 0.0
