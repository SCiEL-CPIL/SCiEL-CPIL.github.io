"""Lennard-Jones cluster N=55: analytic Mackay icosahedron + L-BFGS polish."""
import numpy as np
import time

N = 55

# EVOLVE-BLOCK-START

def _lj_energy_and_grad(pos):
    """Compute total LJ energy and gradient for a configuration."""
    n = pos.shape[0]
    # Compute all pairwise displacements and distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero on diagonal
    r = np.sqrt(r2)
    
    # LJ: 4*(r^-12 - r^-6)
    r6 = r ** (-6)
    r12 = r6 * r6
    pair_energy = 4.0 * (r12 - r6)
    # Set diagonal to 0
    np.fill_diagonal(pair_energy, 0.0)
    total_energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: d/dr [4(r^-12 - r^-6)] = 4(-12 r^-13 + 6 r^-7) = -48 r^-13 + 24 r^-7
    # dE/dpos_i = sum_j 4*(-12 r_ij^-13 + 6 r_ij^-7) * (pos_i - pos_j)/r_ij
    # = sum_j (-48 r_ij^-14 + 24 r_ij^-8) * (pos_i - pos_j)
    # factor = 4 * (-12*r^-13 + 6*r^-7) / r = 4*(-12*r^-14 + 6*r^-8)
    factor = 4.0 * (-12.0 * r2 ** (-7.0) + 6.0 * r2 ** (-4.0))
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return total_energy, grad


def _construct_mackay_icosahedron():
    """
    Construct the two-shell Mackay icosahedron for 55 atoms.
    
    Shell 1: 13 atoms (1 center + 12 vertices of an icosahedron)
    Shell 2: 42 atoms (12 vertices + 30 edge-midpoints of the icosahedron)
    
    The icosahedron has 12 vertices, 30 edges, 20 faces.
    Mackay icosahedron shells:
      - Inner shell (n=1): 1 + 12 = 13 atoms
      - Outer shell (n=2): 12 + 30 = 42 atoms
    Total: 55 atoms.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    
    # 12 vertices of a regular icosahedron (centered at origin)
    # Standard icosahedron vertices: (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize so that the nearest-neighbor distance (edge length) is some value
    # The edge length of this icosahedron: distance between (0,1,phi) and (1,phi,0)
    # = sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + 0.618^2 + 2.618^2) = sqrt(1+0.382+6.854) = sqrt(8.236)
    # Actually let's just normalize vertices to unit radius first, then scale.
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    # The icosahedron circumradius R relates to edge length a by: R = a * sqrt(10 + 2*sqrt(5)) / 4
    # For LJ, the optimal nearest-neighbor distance is 2^(1/6) ≈ 1.12246
    # In the Mackay icosahedron, the inner shell has 12 atoms at the icosahedron vertices.
    # The nearest neighbors of the center atom are the 12 vertices, so we want that distance = 2^(1/6).
    
    # Actually, in a Mackay icosahedron, the structure is built from icosahedral shells.
    # Shell 1: center + 12 at icosahedron vertices (distance d1 from center)
    # Shell 2: 12 at icosahedron vertices (distance d2 from center) + 30 at edge midpoints (distance d2 from center)
    # The ratio d2/d1 is related to the icosahedral geometry.
    
    # For a regular icosahedron with edge length a:
    #   circumradius R = a * sqrt(10 + 2*sqrt(5)) / 4
    #   mid-edge distance from center = a * sqrt(10 + 2*sqrt(5)) / 4 * cos(angle) ... 
    #   Actually, edge midpoint distance from center: for an edge, the midpoint is at distance
    #   R * cos(theta/2) where theta is the central angle subtended by the edge.
    #   The central angle for adjacent vertices: cos(theta) = 1/sqrt(5) ... let me compute directly.
    
    # Let's just use the known structure. The two-shell Mackay icosahedron has:
    # - 13 atoms in the inner icosahedron (center + 12 vertices)
    # - 42 atoms in the outer shell: 12 at the same angular positions as inner vertices but further out,
    #   and 30 at the angular positions of edge midpoints but further out.
    # 
    # The radii ratio is determined by the icosahedral geometry. For the Mackay icosahedron,
    # the outer shell radius / inner shell radius = (1 + phi) / 1 ... actually it's more complex.
    #
    # Let me use a simpler approach: construct the icosahedron with the 12 vertices at distance r1,
    # and the 30 edge midpoints also at distance r1 (they lie on the same sphere as vertices for a regular icosahedron? No, they don't).
    #
    # Actually, for a regular icosahedron, the 12 vertices are on a sphere of radius R, and the 30 edge midpoints
    # are on a sphere of radius R_mid < R. So they're not coplanar with the vertices.
    #
    # The Mackay icosahedron shells are:
    #   Shell 1: 1 + 12 (center + icosahedron vertices)
    #   Shell 2: 12 + 30 (icosahedron vertices scaled + edge midpoints scaled)
    # The scaling factor for shell 2 vs shell 1 is such that the nearest-neighbor distances are optimal.
    #
    # A practical approach: place all 55 atoms and let L-BFGS relax. The key is to get the topology right.
    
    # Inner shell: center at origin, 12 vertices of icosahedron
    # Let's set the inner icosahedron vertex radius to r1
    # The edge length of the inner icosahedron is a1 = 2 * r1 * sin(theta/2) where theta is the central angle between adjacent vertices.
    
    # Central angle between adjacent icosahedron vertices:
    # cos(theta) = dot product of two adjacent unit vertices
    # Take v1 = (0, 1, phi)/|.| and v2 = (1, phi, 0)/|.|
    # dot = 0*1 + 1*phi + phi*0 = phi
    # |v1| = |v2| = sqrt(1 + phi^2)
    # cos(theta) = phi / (1 + phi^2)
    phi2 = phi * phi
    cos_theta = phi / (1.0 + phi2)
    theta = np.arccos(cos_theta)
    
    # For the Mackay icosahedron, a good starting radius for the inner shell:
    # We want the nearest-neighbor distance (center to vertex) to be around 2^(1/6) ≈ 1.122
    r1 = 1.122462048309373  # 2^(1/6)
    
    # Inner shell: 13 atoms
    inner = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        verts_unit * r1
    ])
    
    # Outer shell: 42 atoms
    # The 12 outer vertices are at the same angular positions as the 12 inner vertices, but at a larger radius.
    # The 30 outer edge-midpoint atoms are at the angular positions of the icosahedron edge midpoints, at the same larger radius.
    
    # Compute the 30 edge midpoints (unit vectors)
    # Edges of the icosahedron: connect vertices that are at distance a (edge length) apart.
    # For unit-radius icosahedron, edge length a_unit = 2 * sin(theta/2)
    a_unit = 2.0 * np.sin(theta / 2.0)
    
    # Find all pairs of vertices that are edges (distance ≈ a_unit)
    edge_mids_unit = []
    for i in range(12):
        for j in range(i + 1, 12):
            dist = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(dist - a_unit) < 1e-9:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-12:
                    edge_mids_unit.append(mid / mid_norm)
    edge_mids_unit = np.array(edge_mids_unit)
    
    # There should be 30 edge midpoints
    assert len(edge_mids_unit) == 30, f"Expected 30 edge midpoints, got {len(edge_mids_unit)}"
    
    # Outer shell radius: for the Mackay icosahedron, the ratio is approximately
    # r2/r1 = (1 + phi) ≈ 1.618? Let's use a reasonable estimate and let L-BFGS fix it.
    # Actually, the Mackay icosahedron shell radii follow a pattern. For n=1,2:
    # The ratio is related to the icosahedral lattice. A common approximation:
    # r2 = r1 * (1 + 1/phi) = r1 * phi ≈ 1.618 * r1
    # But let's be more careful. The Mackay icosahedron is a quasicrystalline structure.
    # For now, use r2 = r1 * 1.618 and let L-BFGS optimize.
    r2 = r1 * phi  # ≈ 1.816
    
    # Outer shell: 12 vertices at r2, 30 edge midpoints at r2
    outer = np.vstack([
        verts_unit * r2,
        edge_mids_unit * r2
    ])
    
    pos = np.vstack([inner, outer])
    return pos


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin from 150s limit
    
    # Construct the Mackay icosahedron
    pos = _construct_mackay_icosahedron()
    
    # Verify minimum distance constraint
    def _min_dist(p):
        d = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        r = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(r, np.inf)
        return np.min(r)
    
    # If any atoms are too close, nudge them apart
    md = _min_dist(pos)
    if md < 0.5:
        # Simple fix: scale up
        scale = 0.5 / md * 1.01
        pos = pos * scale
        md = _min_dist(pos)
    
    # Use L-BFGS to relax
    from scipy.optimize import minimize
    
    def _objective_and_grad(flat):
        p = flat.reshape(N, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
    
    best_pos = pos.copy()
    best_energy = _lj_energy_and_grad(pos)[0]
    
    # L-BFGS relaxation
    try:
        result = minimize(
            _objective_and_grad,
            pos.ravel(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.success or result.fun < best_energy:
            cand = result.x.reshape(N, 3)
            # Check minimum distance
            md_cand = _min_dist(cand)
            if md_cand >= 0.5:
                best_energy = result.fun
                best_pos = cand
            else:
                # If constraint violated, keep original
                pass
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        return best_pos
    
    # Try a second, tighter polish if time permits
    try:
        result2 = minimize(
            _objective_and_grad,
            best_pos.ravel(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if result2.fun < best_energy:
            cand = result2.x.reshape(N, 3)
            md_cand = _min_dist(cand)
            if md_cand >= 0.5:
                best_energy = result2.fun
                best_pos = cand
    except Exception:
        pass
    
    return best_pos

# EVOLVE-BLOCK-END
