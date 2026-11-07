"""Rewritten program: systematic multi-structure initialization + shake-relax refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_fragment(n, spacing=1.0):
    """Generate n points from an FCC lattice, centered, sorted by radius."""
    pts = []
    # FCC lattice points: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
    # Generate a large FCC grid and pick closest n to origin
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    fcc_pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC primitive vectors: a1=(0,0.5,0.5), a2=(0.5,0,0.5), a3=(0.5,0.5,0)
                # Or simpler: all integer (i,j,k) where i+j+k even (BCC-like) OR
                # Standard FCC: points at (i,j,k) where i,j,k all integers, plus face centers
                # Let's use: generate all (i,j,k) in range, keep those where i+j+k is even (this gives BCC)
                # Actually for FCC: lattice points at (i,j,k) where i,j,k integers AND (i+j+k) even gives BCC.
                # FCC: use the fact that FCC = all (i,j,k) with i+j+k even is wrong.
                # Correct FCC: generate points at integer coords where i+j+k is even -> that's BCC.
                # Let me just use a simple approach: generate a cubic grid, take FCC subset.
                pass
    # Simpler: generate FCC by taking all (i,j,k) where i,j,k are integers and i+j+k is even
    # This actually gives a BCC lattice. For FCC, use:
    # FCC lattice vectors: a1=(a/2,a/2,0), a2=(a/2,0,a/2), a3=(0,a/2,a/2)
    # Points: n1*a1 + n2*a2 + n3*a3 for integers n1,n2,n3
    fcc_pts = []
    for n1 in range(-max_coord, max_coord + 1):
        for n2 in range(-max_coord, max_coord + 1):
            for n3 in range(-max_coord, max_coord + 1):
                x = (n1 + n2) * 0.5
                y = (n2 + n3) * 0.5
                z = (n3 + n1) * 0.5
                fcc_pts.append((x, y, z))
    fcc_pts = np.array(fcc_pts) * spacing
    # Center
    fcc_pts -= np.mean(fcc_pts, axis=0)
    # Sort by radius
    radii = np.linalg.norm(fcc_pts, axis=1)
    idx = np.argsort(radii)
    return fcc_pts[idx[:n]]


def _hcp_fragment(n, spacing=1.0):
    """Generate n points from HCP lattice."""
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    max_layer = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        # Hexagonal packing in xy
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                if layer % 2 == 0:
                    x = (i + 0.5 * (j % 2)) * a
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = (i + 0.5 * ((j + 1) % 2)) * a
                    y = j * a * np.sqrt(3) / 2
                pts.append((x, y, z))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    radii = np.linalg.norm(pts, axis=1)
    idx = np.argsort(radii)
    return pts[idx[:n]]


def _icosahedral_mackay(n, spacing=1.0):
    """Generate points on concentric icosahedral shells (Mackay structure)."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Mackay shells: radii 1, sqrt(1+1/phi^2)*... actually standard Mackay:
    # Shell 1: 12 vertices (icosahedron), radius 1
    # Shell 2: 30 vertices (edge midpoints of icosahedron), radius sqrt(1+1/phi^2)
    # Shell 3: 20 vertices (face centers), radius sqrt(1+1/(3*phi^2))... 
    # Actually Mackay 55 = 12 + 42? No: 12+30+12=54, or 12+30+20=62.
    # Standard Mackay 55: inner icosahedron (12) + outer shell of 42? 
    # Let me use: shell radii proportional to 1, sqrt(1+1/phi^2), and generate points on each.
    
    shell_radii = [1.0, np.sqrt(1.0 + 1.0/phi**2), np.sqrt(1.0 + 1.0/(3.0*phi**2))]
    
    # Generate points on each shell by projecting icosahedron vertices and their combinations
    all_pts = []
    for r in shell_radii:
        # Use icosahedron vertices scaled
        shell_pts = verts * r
        all_pts.append(shell_pts)
        # Also add edge midpoints for denser shells
        # For simplicity, just use vertices for now
    
    # Better: generate many candidate points on spheres and pick closest n to origin
    # Use a spherical Fibonacci-like distribution
    candidates = []
    for r in [0.5, 0.7, 1.0, 1.2, 1.5]:
        # Fibonacci sphere
        N = 100
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(N):
            z = 1.0 - 2.0 * i / (N - 1)
            radius_xy = np.sqrt(1.0 - z*z)
            theta = golden_angle * i
            x = np.cos(theta) * radius_xy
            y = np.sin(theta) * radius_xy
            candidates.append([x * r, y * r, z * r])
    candidates = np.array(candidates) * spacing
    candidates -= np.mean(candidates, axis=0)
    radii = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(radii)
    return candidates[idx[:n]]


def _spherical_shells(n, spacing=1.0):
    """Concentric spherical shells with Fibonacci distribution."""
    all_pts = []
    # Try different numbers of shells
    for n_shells in [2, 3, 4]:
        pts = []
        for shell_i in range(n_shells):
            r = spacing * (shell_i + 1) * 0.8
            n_shell = max(1, int(n * (r**3) / (n_shells * 0.8**3 * n_shells**3)))
            n_shell = min(n_shell, n - len(pts))
            if n_shell <= 0:
                continue
            N = n_shell
            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            for i in range(N):
                z = 1.0 - 2.0 * i / max(N - 1, 1)
                radius_xy = np.sqrt(1.0 - z*z)
                theta = golden_angle * i
                x = np.cos(theta) * radius_xy
                y = np.sin(theta) * radius_xy
                pts.append([x * r, y * r, z * r])
        if len(pts) >= n:
            all_pts.append(np.array(pts[:n]))
        elif len(pts) > 0:
            # Pad with small random offsets
            pts_arr = np.array(pts)
            while len(pts_arr) < n:
                p = pts_arr[np.random.randint(len(pts_arr))] + np.random.normal(0, 0.05, 3)
                pts_arr = np.vstack([pts_arr, [p]])
            all_pts.append(pts_arr[:n])
    if not all_pts:
        # Fallback: single shell
        N = n
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        r = spacing * n**(1/3) * 0.5
        pts = []
        for i in range(N):
            z = 1.0 - 2.0 * i / max(N - 1, 1)
            radius_xy = np.sqrt(1.0 - z*z)
            theta = golden_angle * i
            x = np.cos(theta) * radius_xy
            y = np.sin(theta) * radius_xy
            pts.append([x * r, y * r, z * r])
        all_pts.append(np.array(pts))
    return all_pts


def _random_compact(n, spacing=0.72, seed=42):
    """Random points with minimum distance, compact."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.6 * spacing
    attempts = 0
    while len(points) < n and attempts < n * 1000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't get enough, relax min distance
    if len(points) < n:
        min_dist = 0.4 * spacing
        points = []
        attempts = 0
        while len(points) < n and attempts < n * 5000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    # If still not enough, just add random
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _shake_and_relax(positions, potential, kind, n_shakes=5, shake_amp=0.05):
    """Shake positions slightly and re-relax, return best."""
    best_e, best_p = sutton_chen_energy_gradient(positions, kind)
    best_e, best_p = best_e, positions
    rng = np.random.default_rng(123)
    for _ in range(n_shakes):
        perturbed = positions + rng.normal(0, shake_amp, positions.shape)
        e, p = relax(perturbed, potential, kind)
        if e < best_e:
            best_e, best_p = e, p
    return best_e, best_p


def construct(instance):
    """Multi-structure initialization: FCC, HCP, icosahedral, spherical shells, random."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    spacing = 0.72  # typical SC spacing
    
    # Generate diverse starting structures
    starts = []
    
    # 1. FCC fragment
    try:
        fcc = _fcc_fragment(size, spacing)
        if len(fcc) == size and np.all(np.isfinite(fcc)):
            starts.append(fcc)
    except Exception:
        pass
    
    # 2. HCP fragment
    try:
        hcp = _hcp_fragment(size, spacing)
        if len(hcp) == size and np.all(np.isfinite(hcp)):
            starts.append(hcp)
    except Exception:
        pass
    
    # 3. Icosahedral/Mackay
    try:
        ico = _icosahedral_mackay(size, spacing)
        if len(ico) == size and np.all(np.isfinite(ico)):
            starts.append(ico)
    except Exception:
        pass
    
    # 4. Spherical shells (multiple variants)
    try:
        shells = _spherical_shells(size, spacing)
        for s in shells:
            if len(s) == size and np.all(np.isfinite(s)):
                starts.append(s)
    except Exception:
        pass
    
    # 5. Random compact (multiple seeds)
    for seed in [42, 123, 456, 789, 1000]:
        try:
            rnd = _random_compact(size, spacing, seed)
            if len(rnd) == size and np.all(np.isfinite(rnd)):
                starts.append(rnd)
        except Exception:
            pass
    
    # Relax all starts, keep best
    best_e = np.inf
    best_p = None
    for i, start in enumerate(starts):
        try:
            e, p = relax(start, potential, kind)
            if np.isfinite(e) and e < best_e:
                best_e = e
                best_p = p
        except Exception:
            continue
    
    if best_p is None:
        # Fallback
        start = _random_compact(size, spacing, 42)
        best_e, best_p = relax(start, potential, kind)
    
    # Shake and re-relax from best
    try:
        e2, p2 = _shake_and_relax(best_p, potential, kind, n_shakes=8, shake_amp=0.08)
        if np.isfinite(e2) and e2 < best_e:
            best_e, best_p = e2, p2
    except Exception:
        pass
    
    # Final validation
    if not np.all(np.isfinite(best_p)):
        # Reset
        start = _random_compact(size, spacing, 42)
        best_e, best_p = relax(start, potential, kind)
    
    # Check minimum distance
    from itertools import combinations
    min_d = np.inf
    for i, j in combinations(range(len(best_p)), 2):
        d = np.linalg.norm(best_p[i] - best_p[j])
        if d < min_d:
            min_d = d
    if min_d < 0.35:
        # Perturb and re-relax
        rng = np.random.default_rng(999)
        for _ in range(10):
            perturbed = best_p + rng.normal(0, 0.1, best_p.shape)
            try:
                e, p = relax(perturbed, potential, kind)
                # Check min distance
                md = np.inf
                for i, j in combinations(range(len(p)), 2):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < md:
                        md = d
                if md >= 0.35 and np.isfinite(e) and e < best_e:
                    best_e, best_p = e, p
                    break
            except Exception:
                continue
    
    return best_p
# EVOLVE-BLOCK-END
