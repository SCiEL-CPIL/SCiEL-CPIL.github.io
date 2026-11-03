<<<<<<< SEARCH
# ---------- Support function convex polygon parameterization ----------
def _support_angles():
    """
    Fixed non-uniform angles for the 14 support function radii.
    Clustered near the 45-degree diagonals to better resolve small triangles.
    """
    # Base: 14 equally spaced angles
    base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Add clustering near diagonals (pi/4, 3pi/4, 5pi/4, 7pi/4)
    # Perturb angles slightly toward diagonals
    diagonals = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    for d in diagonals:
        # Find closest base angle and pull it slightly
        idx = np.argmin(np.abs(base - d))
        base[idx] = base[idx] * 0.7 + d * 0.3
    # Ensure angles are sorted and unique
    base = np.sort(base)
    # Re-normalize to cover full circle
    if base[-1] - base[0] < 2 * np.pi * 0.9:
        base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return base


_SUPPORT_ANGLES = _support_angles()


def _decode_support(radii):
    """
    Decode 14 radial values to 14 points on a convex polygon.
    radii: shape (14,)
    Points are at angles _SUPPORT_ANGLES with the given radii.
    """
    r = np.maximum(radii, 0.05)
    x = r * np.cos(_SUPPORT_ANGLES)
    y = r * np.sin(_SUPPORT_ANGLES)
    pts = np.column_stack([x, y])
    pts -= pts.mean(axis=0)
    return pts


def _objective_support(radii):
    pts = _decode_support(radii)
    return _objective_hard(pts.flatten())


def _objective_support_soft(radii, beta=80.0):
    pts = _decode_support(radii)
    return _objective_soft(pts.flatten(), beta)


def _generate_support_inits():
    """Generate diverse initial radial configurations for support function."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular polygon (all radii equal)
    inits.append(np.ones(N).copy())
    
    # 2. Elliptical radii
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.3, 0.7), (0.7, 1.3), (1.1, 0.9)]:
        r = np.sqrt(rx**2 * np.cos(_SUPPORT_ANGLES)**2 + ry**2 * np.sin(_SUPPORT_ANGLES)**2)
        inits.append(r.copy())
    
    # 3. Perturbed regular polygon
    for js in [0.05, 0.1, 0.15, 0.2]:
        r = np.ones(N) + rng.normal(0, js, size=N)
        r = np.maximum(r, 0.3)
        inits.append(r.copy())
    
    # 4. Random radii
    for seed in range(15):
        rs = np.random.default_rng(seed * 13 + 7)
        r = rs.uniform(0.6, 1.4, N)
        inits.append(r.copy())
    
    # 5. Alternating large/small radii
    for seed in range(5):
        rs = np.random.default_rng(seed * 23 + 11)
        r = np.ones(N)
        r[::2] = rs.uniform(0.8, 1.2)
        r[1::2] = rs.uniform(0.6, 1.0)
        inits.append(r.copy())
    
    # 6. Larger perturbations
    for seed in range(10):
        rs = np.random.default_rng(seed * 31 + 17)
        r = rs.uniform(0.4, 1.6, N)
        inits.append(r.copy())
    
    return inits
=======
# ---------- Fourier Support Function Parameterization ----------
def _decode_fourier(params):
    """
    Decode 9-dim parameters to 14 points.
    params: [R0, a1, b1, a2, b2, a3, b3, a4, b4]
    h(theta) = R0 + sum(a_k cos(k theta) + b_k sin(k theta))
    We sample h at 14 equally spaced angles to get vertices.
    """
    R0 = params[0]
    coeffs = params[1:]
    
    # Ensure convexity and positivity: R0 must be large enough
    # h''(theta) = -sum(k^2 (a_k cos + b_k sin))
    # For safety, we clamp R0 to be at least sum(k^2 * |coeffs|) * 1.1 + 0.1
    # But R0 is a variable. We can just use a soft penalty or clamp in objective.
    # Here we just decode. If it's not convex, ConvexHull will handle it (taking hull).
    # However, the task says "Place 14 points in a planar convex region".
    # If the points are not in convex position, the hull area is used for normalization.
    # The score is min_triangle_area / hull_area.
    # If points are inside the hull, min_triangle_area might be smaller than if they were on the hull.
    # So we want the points to BE the hull vertices.
    # To ensure this, we should ensure h(theta) + h''(theta) > 0.
    
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = np.full(N, R0)
    for k in range(1, 5):
        ak = coeffs[2*(k-1)]
        bk = coeffs[2*(k-1)+1]
        h += ak * np.cos(k * theta) + bk * np.sin(k * theta)
        
    # Ensure h is positive
    h = np.maximum(h, 0.1)
    
    x = h * np.cos(theta)
    y = h * np.sin(theta)
    pts = np.column_stack([x, y])
    pts -= pts.mean(axis=0)
    return pts


def _objective_fourier(params):
    pts = _decode_fourier(params)
    return _objective_hard(pts.flatten())


def _generate_fourier_inits():
    """Generate diverse initial Fourier coefficient sets."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Circle (R0=1, others 0)
    inits.append(np.array([1.0, 0,0, 0,0, 0,0, 0,0]))
    
    # 2. Ellipses (k=1 dominant)
    for a1 in [0.1, 0.2, 0.3]:
        for b1 in [0.1, 0.2, 0.3]:
            inits.append(np.array([1.0, a1, b1, 0,0, 0,0, 0,0]))
            
    # 3. Squarish (k=2 dominant)
    for a2 in [0.05, 0.1, 0.15]:
        for b2 in [0.05, 0.1, 0.15]:
            inits.append(np.array([1.0, 0,0, a2, b2, 0,0, 0,0]))
            
    # 4. Random small perturbations
    for seed in range(20):
        rs = np.random.default_rng(seed)
        p = np.zeros(9)
        p[0] = 1.0
        p[1:] = rs.normal(0, 0.1, 8)
        inits.append(p)
        
    # 5. Larger random perturbations
    for seed in range(10):
        rs = np.random.default_rng(seed + 100)
        p = np.zeros(9)
        p[0] = 1.0
        p[1:] = rs.normal(0, 0.2, 8)
        inits.append(p)
        
    return inits
>>>>>>> REPLACE

