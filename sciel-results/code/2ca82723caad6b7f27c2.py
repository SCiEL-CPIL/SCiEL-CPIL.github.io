"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms."""
    n = len(pos)
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=-1)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r6 = r2**(-3)
    r12 = r6**2
    # Energy: 4 * sum(r^-12 - r^-6) over pairs (each pair counted once)
    energy = 4.0 * np.sum(r12 - r6) - 4.0 * (1.0 - 1.0)  # diagonal is 0 contribution
    # Actually diagonal: r2=1, r6=1, r12=1, so r12-r6=0, fine
    energy = 4.0 * np.sum(r12 - r6)
    # But we double-counted pairs, so divide by 2
    energy /= 2.0
    # Gradient: dE/dr_i = 4 * sum_j (12*r_i^-13 - 6*r_i^-7) * (r_i - r_j)
    # More carefully: E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dx_i = 4 * sum_{j != i} (-12*r_ij^-14 + 6*r_ij^-8) * (x_i - x_j)
    # = 4 * sum_{j != i} (-12/r_ij^2 * r_ij^-12 + 6/r_ij^2 * r_ij^-6) * (x_i - x_j)
    # = 4 * sum_{j != i} (-12*r6[j,i] + 6*r6[j,i]**2 / r2[j,i]) * (x_i - x_j)
    # Let me use a cleaner formulation
    # dE/dx_i = 4 * sum_{j != i} (12*r_ij^-14 - 6*r_ij^-8) * (x_i - x_j)
    # Note: 12*r^-14 = 12*r^-12/r^2, 6*r^-8 = 6*r^-6/r^2
    # So: 4 * sum_{j != i} (12*r6/r2 - 6*r6**2/r2) * (x_i - x_j)
    #    = 4/r2 * (12*r6 - 6*r12) * (x_i - x_j) summed over j
    # Using matrix form:
    # factor[i,j] = 4 * (12*r6[i,j] - 6*r12[i,j]) / r2[i,j] for i != j
    # grad[i] = sum_j factor[i,j] * (pos[i] - pos[j])
    r2_safe = np.where(r2 > 1e-10, r2, 1.0)
    r6_safe = r2_safe**(-3)
    r12_safe = r6_safe**2
    # factor matrix: 4*(12*r6 - 6*r12)/r2, with diagonal = 0
    factor = 4.0 * (12.0 * r6_safe - 6.0 * r12_safe) / r2_safe
    np.fill_diagonal(factor, 0.0)
    # grad[i] = sum_j factor[i,j] * (pos[i] - pos[j])
    # = pos[i] * sum_j factor[i,j] - sum_j factor[i,j] * pos[j]
    # But simpler: grad = factor @ (-pos) + pos * factor.sum(axis=1, keepdims=True)
    # Actually: sum_j factor[i,j]*(pos[i]-pos[j]) = pos[i]*sum_j factor[i,j] - sum_j factor[i,j]*pos[j]
    # Let me just compute directly
    grad = np.zeros_like(pos)
    for i in range(n):
        grad[i] = np.sum(factor[i, :] * (pos[i] - pos), axis=0)
    return energy, grad


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Construct the FCC truncated octahedron for LJ38
    # Structure: 1 center + 12 face centers + 24 edge centers
    # In a truncated octahedron, the vertices are at:
    # Permutations of (0, ±1, ±2), (±1, ±2, 0), (±2, 0, ±1)
    # But we need to place atoms at specific positions
    
    # The known global minimum for LJ38 is a truncated octahedron
    # Let me construct it using the known coordinates
    
    # Truncated octahedron vertices (24 vertices) + 12 face centers + 1 center = 37
    # Wait, we need 38 atoms. Let me reconsider.
    
    # Actually, for LJ38, the structure is:
    # - 1 atom at center
    # - 12 atoms at octahedral positions (face centers of cube)
    # - 24 atoms at positions corresponding to truncated octahedron vertices
    
    # Let me use a different approach: start from a reasonable structure and relax
    
    # Method 1: FCC lattice fragment
    def fcc_structure():
        """Generate FCC-like structure for 38 atoms."""
        # FCC lattice vectors
        a = 1.12  # approximate equilibrium spacing
        # Generate FCC points
        points = []
        # Center
        points.append(np.array([0.0, 0.0, 0.0]))
        # 12 nearest neighbors in FCC (octahedral + tetrahedral)
        # Octahedral: (±a/2, 0, 0), (0, ±a/2, 0), (0, 0, ±a/2) -> 6
        # Tetrahedral: (±a/4, ±a/4, ±a/4) -> 8
        # That's 1+6+8 = 15, need more
        
        # Let me use a simpler approach: cubic lattice with FCC-like spacing
        # and then relax
        pts = []
        # 3x3x3 grid = 27, plus 11 more
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    pts.append([(i-1)*a, (j-1)*a, (k-1)*a])
        # Add 11 more atoms at FCC positions
        # Face centers of the 3x3x3 cube
        # Top/bottom: (0,0,±1.5a), (1,0,±1.5a), etc.
        a_half = a / 2
        extra = [
            [a_half, a_half, a_half],
            [-a_half, a_half, a_half],
            [a_half, -a_half, a_half],
            [a_half, a_half, -a_half],
            [-a_half, -a_half, a_half],
            [-a_half, a_half, -a_half],
            [a_half, -a_half, -a_half],
            [-a_half, -a_half, -a_half],
            [1.5*a, 0, 0],
            [-1.5*a, 0, 0],
            [0, 1.5*a, 0],
        ]
        for e in extra:
            pts.append(e)
        
        pts = np.array(pts[:38], dtype=float)
        # Center it
        pts -= pts.mean(axis=0)
        return pts
    
    def icosahedral_structure():
        """Generate icosahedral-like structure for 38 atoms."""
        # Icosahedron: 12 vertices, 1 center, plus shells
        a = 1.12
        # 12 icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = []
        for perm in [(0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
                     (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
                     (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)]:
            ico_verts.append(perm)
        ico_verts = np.array(ico_verts, dtype=float)
        # Normalize to radius a
        norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
        ico_verts = ico_verts / norms * a
        
        pts = [np.array([0.0, 0.0, 0.0])]  # center
        pts.extend([v for v in ico_verts])  # 12 icosahedron vertices
        # Add more shells
        # 20 face centers of icosahedron
        # And 30 edge midpoints
        # For simplicity, add random jittered positions
        rng = np.random.RandomState(42)
        for _ in range(25):  # need 38 total, have 13, need 25 more
            # Random position on sphere of radius ~1.5*a
            theta = rng.uniform(0, 2*np.pi)
            z = rng.uniform(-1, 1)
            r = 1.5 * a
            x = np.sqrt(1-z*z) * np.cos(theta) * r
            y = np.sqrt(1-z*z) * np.sin(theta) * r
            pts.append(np.array([x, y, z]))
        
        pts = np.array(pts[:38], dtype=float)
        pts -= pts.mean(axis=0)
        return pts
    
    def random_structure(seed):
        """Generate random structure."""
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-2, 2, size=(38, 3))
        pts -= pts.mean(axis=0)
        return pts
    
    best_pos = None
    best_energy = np.inf
    
    # Try multiple initial configurations
    configs = []
    configs.append(("fcc", fcc_structure()))
    configs.append(("ico", icosahedral_structure()))
    for s in range(5):
        configs.append((f"rand{s}", random_structure(s)))
    
    for name, pos in configs:
        if time_left() < 15:
            break
        
        # Ensure minimum separation
        pos = pos.copy()
        
        # Relax using L-BFGS
        try:
            result = minimize(
                lj_energy_and_grad,
                pos,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000}
            )
            if result.success or result.fun < best_energy:
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.copy()
        except Exception:
            pass
    
    # If we didn't find anything good, try a more systematic approach
    if best_pos is None or best_energy > -100:
        # Try a simple cubic lattice and relax
        a = 1.12
        pts = []
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    pts.append([(i-1.5)*a, (j-1.5)*a, (k-1)*a])
        pts = np.array(pts[:38], dtype=float)
        pts -= pts.mean(axis=0)
        
        if time_left() > 10:
            try:
                result = minimize(
                    lj_energy_and_grad,
                    pts,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 20000}
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.copy()
            except Exception:
                pass
    
    # Final check: ensure atoms are at least 0.5 apart
    if best_pos is not None:
        # Check minimum distance
        rij = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(rij**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 0.5:
            # Scale up if needed
            scale = 0.5 / min_dist * 1.01
            best_pos = best_pos * scale
    
    if best_pos is None:
        # Fallback: return the simple cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
# EVOLVE-BLOCK-END
