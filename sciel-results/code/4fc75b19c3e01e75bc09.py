def get_axes(vertices):
    """Get normal vectors for the edges of a polygon."""
    axes = []
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # Normal vector
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            axes.append(normal / norm)
    return np.array(axes)

def project(vertices, axis):
    """Project vertices onto an axis."""
    return np.dot(vertices, axis)

def polygons_disjoint(v1, v2, eps=1e-9):
    """
    Check if two convex polygons are disjoint using Separating Axis Theorem.
    Returns True if disjoint, False if overlapping.
    """
    axes = np.vstack([get_axes(v1), get_axes(v2)])
    
    for axis in axes:
        proj1 = project(v1, axis)
        proj2 = project(v2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return True
    return False

def point_in_polygon(pt, vertices):
    """Check if point is inside a convex polygon."""
    n = len(vertices)
    sign = 0
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        cross = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
        if cross > 1e-9:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-9:
            if sign > 0:
                return False
            sign = -1
    return True

def hexagon_inside_hexagon(inner_v, outer_v):
    """Check if inner polygon vertices are all inside outer polygon."""
    for v in inner_v:
        if not point_in_polygon(v, outer_v):
            return False
    return True
