"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient. pos shape (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero on diagonal
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 * r6
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    energy = 2.0 * np.sum(4.0 * (r12 - r6))  # factor 2 because sum over i<j counted twice
    # Actually let's be more careful.
    # E = sum_{i<j} 4*(r_ij^-12 - r_ij^-6)
    # Using the full matrix, sum_{i!=j} 4*(r_ij^-12 - r_ij^-6) / 2
    # = 2 * sum_{i!=j} 4*(r_ij^-12 - r_ij^-6) ... no
    # Let me just compute it properly.
    
    # E = 0.5 * sum_{i != j} 4 * (r_ij^{-12} - r_ij^{-6})
    # = 2 * sum_{i != j} (r_ij^{-12} - r_ij^{-6})
    
    e_pair = 4.0 * (r12 - r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = 0.5 * np.sum(e_pair)
    
    # Gradient: dE/dx_i = sum_{j != i} dE/dx_ij
    # dE/dx_i = -24 * sum_j [2*r_i^-14 * (x_i - x_j) - r_i^-8 * (x_i - x_j)] ... let me derive
    # E_ij = 4*(r^-12 - r^-6)
    # dE_ij/dr = 4*(-12*r^-13 + 6*r^-7) = 24*(r^-7 - 2*r^-13)
    # dE_ij/dx_i = dE_ij/dr * dr/dx_i = 24*(r^-7 - 2*r^-13) * (x_i - x_j)/r
    #            = 24*(r^-8 - 2*r^-14) * (x_i - x_j)
    #            = 24*(r2^-4 - 2*r2^-7) * (x_i - x_j)
    
    # grad[i] = sum_{j != i} 24*(r_ij^-8 - 2*r_ij^-14) * (x_i - x_j)
    
    coeff = 24.0 * (r2 ** -4 - 2.0 * r2 ** -7)
    np.fill_diagonal(coeff, 0.0)
    grad = -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    # Wait, let me check the sign.
    # dE/dx_i = sum_j dE_ij/dx_i
    # dE_ij/dx_i = 24*(r^-8 - 2*r^-14) * (x_i - x_j)
    # So grad[i] = sum_j 24*(r_ij^-8 - 2*r_ij^-14) * (x_i - x_j)
    # diff[i,j] = x_i - x_j
    # So grad[i] = sum_j coeff[i,j] * diff[i,j] where coeff = 24*(r2^-4 - 2*r2^-7)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _make_fcc_truncated_octahedron(n=38):
    """
    Construct FCC truncated octahedron for LJ38.
    The known global minimum for LJ38 is the FCC truncated octahedron.
    Structure: 
    - 1 atom at center
    - 12 atoms at first shell (cuboctahedron vertices)
    - 14 atoms at second shell
    - 11 atoms at third shell
    Actually, the standard LJ38 structure has layers.
    Let me use the known coordinates from the Cambridge Cluster Database approach.
    
    The FCC truncated octahedron for n=38:
    It's a fragment of FCC lattice. FCC has 4 atoms per unit cell.
    Truncated octahedron with 38 atoms:
    Layer structure (along [111]): 
    The standard description uses a specific arrangement.
    
    Let me use a simpler approach: generate FCC lattice points and select 38 closest to center.
    FCC lattice points: (i,j,k) where i+j+k is even, scaled by 1/sqrt(2) for nearest neighbor = 1.
    Actually, for LJ with sigma=1, the equilibrium nearest-neighbor distance is 2^(1/6) ≈ 1.1225.
    So I should scale the FCC lattice so that nearest neighbors are at 2^(1/6).
    
    FCC primitive vectors: a/2*(0,1,1), a/2*(1,0,1), a/2*(1,1,0) where a is lattice constant.
    Nearest neighbor distance = a/sqrt(2).
    We want a/sqrt(2) = 2^(1/6), so a = 2^(1/6)*sqrt(2) = 2^(2/3) ≈ 1.5874.
    
    Let me generate FCC points and pick 38 closest to origin.
    """
    a = 2.0 ** (2.0 / 3.0)  # lattice constant for nearest neighbor = 2^(1/6)
    # FCC points: all (i,j,k) with i+j+k even, in units of a/2
    # Actually, FCC = simple cubic with basis (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # In units of a: positions are (ia + 0, ja + 0, ka + 0), (ia, ja, ka) + (0, a/2, a/2), etc.
    
    points = []
    max_range = 5
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                # FCC basis
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    points.append((x, y, z))
    
    points = np.array(points)
    # Remove duplicates
    points = np.unique(np.round(points, 10), axis=0)
    
    # Sort by distance from origin
    dists = np.sum(points ** 2, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    return points


def _make_icosahedral_structure(n=38):
    """
    Construct an icosahedral-like structure for LJ38.
    Icosahedral shells: 1 + 12 + 20 = 33 for the Mackay icosahedron.
    For 38, we add 5 more atoms.
    """
    points = []
    # Center
    points.append([0.0, 0.0, 0.0])
    
    # Icosahedron vertices (12 atoms) at radius r1
    r1 = 2.0 ** (1.0 / 6.0)  # equilibrium distance
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = []
    for perm in [(0, 1, phi), (0, -1, phi), (1, phi, 0), (-1, phi, 0),
                 (1, phi, 0), (-1, phi, 0), (0, 1, phi), (0, -1, phi),
                 (1, 0, phi), (-1, 0, phi), (1, 0, -phi), (-1, 0, -phi)]:
        pass
    
    # Standard icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    # Scale to r1
    verts = verts * r1
    points.extend(verts.tolist())
    
    # Second shell: 20 atoms at face centers of icosahedron
    # Faces of icosahedron
    faces = [
        [0, 1, 8], [0, 1, 4], [0, 4, 10], [0, 10, 7], [0, 7, 3],
        [1, 0, 8], [1, 8, 11], [1, 11, 5], [1, 5, 4],
        [4, 1, 5], [4, 5, 9], [4, 9, 10],
        [10, 4, 9], [10, 9, 2], [10, 2, 7],
        [7, 10, 2], [7, 2, 6], [7, 6, 3],
        [3, 7, 6], [3, 6, 11], [3, 11, 8]
    ]
    # Compute face centers
    face_centers = []
    for f in faces:
        fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        fc = fc / np.linalg.norm(fc) * (r1 * np.sqrt(5.0 / 3.0))  # approximate radius
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    points.extend(face_centers.tolist())
    
    # We have 1 + 12 + 20 = 33 atoms. Need 5 more.
    # Add 5 atoms at edge midpoints or other positions
    # For simplicity, add 5 atoms at random positions on a sphere of radius ~1.5*r1
    rng = np.random.RandomState(42)
    for _ in range(5):
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z ** 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z] * r1 * 1.5)
    
    return np.array(points[:n])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. FCC truncated octahedron
    fcc = _make_fcc_truncated_octahedron(N)
    inits.append(fcc)
    
    # 2. Icosahedral structure
    ico = _make_icosahedral_structure(N)
    inits.append(ico)
    
    # 3. Random spherical shell configurations
    rng = np.random.RandomState(123)
    for trial in range(50):
        if time.time() > deadline - 30.0:
            break
        # Assign atoms to shells
        n_shells = rng.randint(2, 5)
        radii = np.linspace(0.5, 3.0, n_shells)
        counts = np.zeros(n_shells, dtype=int)
        counts[0] = rng.randint(1, 4)
        remaining = N - counts[0]
        for s in range(1, n_shells):
            if s == n_shells - 1:
                counts[s] = remaining
            else:
                counts[s] = rng.randint(1, remaining + 1)
                remaining -= counts[s]
        
        pos = np.zeros((N, 3))
        idx = 0
        for s in range(n_shells):
            n_s = counts[s]
            if n_s == 0:
                continue
            # Random points on sphere of radius radii[s]
            thetas = rng.uniform(0, 2 * np.pi, n_s)
            z_vals = rng.uniform(-1, 1, n_s)
            r_vals = np.sqrt(np.maximum(0, 1 - z_vals ** 2))
            x = r_vals * np.cos(thetas)
            y = r_vals * np.sin(thetas)
            z = z_vals
            shell_pos = np.column_stack([x, y, z]) * radii[s]
            # Add angular jitter
            shell_pos += rng.normal(0, 0.1, size=shell_pos.shape)
            pos[idx:idx + n_s] = shell_pos
            idx += n_s
        
        inits.append(pos)
    
    # 4. Perturbed FCC
    for trial in range(10):
        if time.time() > deadline - 30.0:
            break
        fcc_pert = fcc + rng.normal(0, 0.1, size=fcc.shape)
        inits.append(fcc_pert)
    
    # 5. Perturbed icosahedron
    for trial in range(10):
        if time.time() > deadline - 30.0:
            break
        ico_pert = ico + rng.normal(0, 0.1, size=ico.shape)
        inits.append(ico_pert)
    
    # Try each initial configuration
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        
        # Ensure minimum distance constraint
        # Simple check: if any pair is too close, perturb
        diff = init[:, np.newaxis, :] - init[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            # Perturb randomly
            init = init + rng.normal(0, 0.2, size=init.shape)
        
        # Local relaxation
        try:
            res = minimize(
                _lj_energy_and_grad,
                init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            e, _ = _lj_energy_and_grad(res.x)
            if e < best_energy:
                best_energy = e
                best_pos = res.x.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: return FCC
        best_pos = fcc
        best_energy = _lj_energy_and_grad(fcc)[0]
    
    return best_pos
# EVOLVE-BLOCK-END
