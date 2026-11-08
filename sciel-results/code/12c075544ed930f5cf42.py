<<<<<<< SEARCH
def minkowski_sum_inradius(v1, v2):
    """
    Compute the inradius of the Minkowski sum of two convex polygons v1 and v2.
    """
    from scipy.spatial import ConvexHull
    sums = []
    for a in v1:
        for b in v2:
            sums.append(a + b)
    sums = np.array(sums)
    
    try:
        hull = ConvexHull(sums)
        vertices = sums[hull.vertices]
    except Exception:
        return 0.0

    min_dist = float('inf')
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        cross = abs(p1[0] * p2[1] - p1[1] * p2[0])
        length = math.sqrt(dx*dx + dy*dy)
        if length > 1e-9:
            dist = cross / length
            if dist < min_dist:
                min_dist = dist
    return min_dist

# Precompute R_min table
# Reduced steps to 36 (10 degrees) to speed up initialization
ALPHA_STEPS = 36 
R_MIN_TABLE = []
v1_base = hex_vertices(0.0)
for i in range(ALPHA_STEPS + 1):
    alpha = (i / ALPHA_STEPS) * (math.pi / 3)
    v2_rot = hex_vertices(-alpha)
    r_min = minkowski_sum_inradius(v1_base, v2_rot)
    R_MIN_TABLE.append(r_min)

def get_r_min(alpha):
    """Interpolate R_min for a given relative angle alpha."""
    alpha = alpha % (math.pi / 3)
    if alpha < 0:
        alpha += math.pi / 3
    idx = alpha / (math.pi / 3) * ALPHA_STEPS
    i0 = int(idx)
    i1 = min(i0 + 1, ALPHA_STEPS)
    frac = idx - i0
    return R_MIN_TABLE[i0] * (1 - frac) + R_MIN_TABLE[i1] * frac
=======
def hexagon_collision_distance(v1, v2):
    """
    Calculate the minimum translation distance to separate two convex hexagons v1 and v2.
    Returns 0.0 if they do not overlap.
    Uses Separating Axis Theorem (SAT).
    """
    # v1 and v2 are 6x2 numpy arrays of vertices in CCW order.
    # We check all 6 normals of v1 and 6 normals of v2.
    max_overlap = 0.0
    n = 6
    
    # Helper to get normals from vertices
    def get_normals(verts):
        normals = []
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            # Edge vector
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # Normal (perpendicular)
            nx = ey
            ny = -ex
            # Normalize
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                nx /= length
                ny /= length
            normals.append((nx, ny))
        return normals

    normals1 = get_normals(v1)
    normals2 = get_normals(v2)
    
    all_normals = normals1 + normals2
    
    for nx, ny in all_normals:
        # Project v1 onto normal
        min1 = float('inf')
        max1 = float('-inf')
        for p in v1:
            proj = p[0]*nx + p[1]*ny
            if proj < min1: min1 = proj
            if proj > max1: max1 = proj
            
        # Project v2 onto normal
        min2 = float('inf')
        max2 = float('-inf')
        for p in v2:
            proj = p[0]*nx + p[1]*ny
            if proj < min2: min2 = proj
            if proj > max2: max2 = proj
            
        # Check for overlap
        if max1 <= min2 or max2 <= min1:
            return 0.0 # Separating axis found, no overlap
        
        # Calculate overlap amount
        # Distance between intervals
        if min2 < min1:
            overlap = min1 - min2
        else:
            overlap = min2 - min1
            
        if overlap > max_overlap:
            max_overlap = overlap
            
    return max_overlap
>>>>>>> REPLACE

