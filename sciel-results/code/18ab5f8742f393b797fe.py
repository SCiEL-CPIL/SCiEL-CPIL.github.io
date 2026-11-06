<<<<<<< SEARCH
def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])  # normalize to unit radius
    return verts


def _make_icosahedral_seed(n, rng, radius_scale=1.0):
    """Generate an icosahedral-like seed with n atoms.
    
    Layer 1: center atom
    Layer 2: 12 icosahedron vertices
    Layer 3+: additional shells with icosahedral symmetry
    """
    ico = _icosahedron_vertices()
    base_spacing = 1.1  # Dzugutov equilibrium spacing
    
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 atoms at icosahedron vertices
    r1 = base_spacing * radius_scale
    for v in ico:
        if len(points) < n:
            points.append(r1 * v)
    
    # Shell 2: 30 atoms at edge midpoints of icosahedron (approx)
    # Use a slightly larger radius with icosahedral directions
    if len(points) < n:
        # Generate more directions by combining icosahedron vertices
        r2 = base_spacing * 1.55 * radius_scale
        # icosahedron has 6 vertices per face, 20 faces -> but we need unique directions
        # Use scaled icosahedron + rotated versions
        angles = [0, np.pi/5, 2*np.pi/5, 3*np.pi/5, 4*np.pi/5]
        for ax in range(3):
            for ang in angles:
                if len(points) >= n:
                    break
                # Rotate icosahedron vertex by angle around each axis
                v = ico[0].copy()
                if ax == 0:
                    v[1], v[2] = v[1]*np.cos(ang) - v[2]*np.sin(ang), v[1]*np.sin(ang) + v[2]*np.cos(ang)
                elif ax == 1:
                    v[0], v[2] = v[0]*np.cos(ang) - v[2]*np.sin(ang), v[0]*np.sin(ang) + v[2]*np.cos(ang)
                else:
                    v[0], v[1] = v[0]*np.cos(ang) - v[1]*np.sin(ang), v[0]*np.sin(ang) + v[1]*np.cos(ang)
                v = v / np.linalg.norm(v)
                points.append(r2 * v)
    
    # Fill remaining with random compact positions
    while len(points) < n:
        r_max = base_spacing * (n ** (1/3)) * 0.8
        p = rng.uniform(-r_max, r_max, 3)
        if np.linalg.norm(p) <= r_max:
            # Check minimum distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55 * base_spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                # Just add with small random perturbation
                base = points[rng.integers(len(points))]
                p = base + rng.normal(0, 0.3, 3)
                points.append(p)
    
    return np.array(points[:n])


def _make_deformed_icosahedral_seed(n, rng, radius_scale=1.0, deform=0.15):
    """Create a deformed icosahedral seed for diversity."""
    seed = _make_icosahedral_seed(n, rng, radius_scale)
    # Apply random deformation
    noise = rng.normal(0, deform, seed.shape)
    seed = seed + noise
    # Re-center
    seed -= seed.mean(axis=0)
    return seed


def _make_random_compact_seed(n, rng, spacing=1.1):
    """Random compact seed with minimum distance constraint."""
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all, add remaining with perturbation
    while len(points) < n:
        if points:
            base = points[rng.integers(len(points))]
            p = base + rng.normal(0, 0.4, 3)
        else:
            p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _relax_and_kick(pos, potential, parameter, rng, max_kicks=3, kick_scale=0.4):
    """Relax, then do kick-and-relax cycles to escape local minima."""
    best_energy = np.inf
    best_pos = pos.copy()
    
    for kick_iter in range(max_kicks + 1):
        try:
            energy, relaxed = relax(pos, potential, parameter)
        except Exception:
            break
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
        
        if kick_iter == max_kicks:
            break
        
        # Kick: add random displacement
        kick = rng.normal(0, kick_scale, relaxed.shape)
        pos = relaxed + kick
    
    return best_energy, best_pos
=======
def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts


def _make_mackay_seed(n, rng):
    """
    Generate a Mackay icosahedral seed.
    For n=13: Center + 12.
    For n=55: Center + 12 + 42.
    For other n: Truncate or extend.
    """
    ico = _icosahedron_vertices()
    # Equilibrium distance approx 1.1
    r1 = 1.1
    r2 = 1.1 * np.sqrt(5.0) / 2.0  # Approx distance for 2nd shell in Mackay
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 atoms
    for v in ico:
        if len(points) < n:
            points.append(r1 * v)
    
    if len(points) < n:
        # Shell 2: 42 atoms in Mackay (12 triangles of 3? No, 20 faces * 2 + 12 vertices? 
        # Actually Mackay 55 has 42 atoms in the second shell.
        # Directions: 30 edge midpoints + 12 face centers? No.
        # Let's use 30 edge midpoints and 12 face centers for 42?
        # Or just generate 42 points on the icosahedron surface using Fibonacci or similar.
        
        # Simple approach: Generate 42 points on a sphere of radius r2
        # Using Icosahedral symmetry: 12 vertices, 30 edge midpoints, 20 face centers = 62 points.
        # We need 42. Let's take 12 vertices (already have), 30 edge midpoints.
        # That's 42.
        
        # Calculate edge midpoints
        # Icosahedron edges: 30.
        # Vertices:
        # 0: [0, 1, phi]
        # ...
        # We can just use a Fibonacci sphere for the remaining n-13 points if n > 13.
        # But Mackay structure is specific.
        
        # Let's use the 30 edge midpoints of the icosahedron.
        # We need to identify edges.
        # Easier: Use a "Golden Spiral" or "Fibonacci" distribution for the outer shell.
        # This ensures good coverage.
        
        num_outer = n - len(points)
        if num_outer > 0:
            # Fibonacci sphere
            phi_golden = np.pi * (3.0 - np.sqrt(5.0))
            for k in range(num_outer):
                y = 1.0 - (k + 0.5) / num_outer * 2.0
                radius_yz = np.sqrt(1.0 - y * y)
                theta = phi_golden * k
                x = np.cos(theta) * radius_yz
                z = np.sin(theta) * radius_yz
                p = r2 * np.array([x, y, z])
                points.append(p)
                
    return np.array(points[:n])


def _make_random_compact_seed(n, rng, spacing=1.1):
    """Random compact seed."""
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.6 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        if points:
            base = points[rng.integers(len(points))]
            p = base + rng.normal(0, 0.4, 3)
        else:
            p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])


def _relax_lbfgsb(pos, potential, parameter):
    """Relax using L-BFGS-B with Dzugutov gradient."""
    try:
        res = minimize(
            _dzugutov_energy,
            pos,
            jac=_dzugutov_gradient,
            method='L-BFGS-B',
            options={'maxiter': 50, 'maxls': 20, 'ftol': 1e-8}
        )
        return res.fun, res.x
    except Exception:
        return np.inf, pos
>>>>>>> REPLACE
