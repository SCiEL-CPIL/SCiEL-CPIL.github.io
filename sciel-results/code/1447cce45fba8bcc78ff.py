def calc_min_area_fast(points):
    """
    Vectorized calculation of minimum triangle area.
    points: (N, 2)
    """
    # Generate indices for all combinations of 3
    # N=11, 165 triangles.
    # We can use itertools.combinations but it's slow in Python.
    # Since N is small, we can pre-compute indices.
    
    # Pre-compute indices once globally?
    # Let's do it inside for clarity, or use a static array.
    
    # Static indices for N=11
    idx = np.array(list(combinations(range(11), 3)))
    # But N is fixed to 11 in the task.
    
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)
