"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# ---------- Icosahedron geometry ----------
def _build_icosahedron():
    """Return unit vectors to the 12 vertices, 30 edge-midpoints, 12 face-centers of a
    regular icosahedron centered at origin. Returns (verts, edges, faces) each normalized."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of icosahedron (cyclic permutations of (0, +-1, +-phi))
    verts = []
    base = [(0.0, 1.0, phi), (0.0, -1.0, phi), (0.0, 1.0, -phi), (0.0, -1.0, -phi),
            (1.0, phi, 0.0), (-1.0, phi, 0.0), (1.0, -phi, 0.0), (-1.0, -phi, 0.0),
            (phi, 0.0, 1.0), (-phi, 0.0, 1.0), (phi, 0.0, -1.0), (-phi, 0.0, -1.0)]
    verts = np.array(base, dtype=float)
    # normalize
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    # edges: pairs of vertices that are nearest neighbors. Compute all pairwise distances,
    # the 12 nearest-neighbor distances per vertex define edges; total 30 unique edges.
    D = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    # smallest off-diagonal distance
    np.fill_diagonal(D, np.inf)
    dmin = D.min()
    # edges are pairs with distance ~ dmin
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - dmin) < 1e-8:
                edges.append((i, j))
    # edge-midpoints
    edges_vec = []
    for (i, j) in edges:
        m = (verts[i] + verts[j]) / 2.0
        m = m / np.linalg.norm(m)
        edges_vec.append(m)
    edges_vec = np.array(edges_vec)

    # faces: each face is a triangle of 3 mutually-nearest vertices.
    # For each vertex, its 5 neighbors form a pentagon; faces are (v, n1, n2) where n1,n2 adjacent in pentagon.
    # Simpler: enumerate all triples (i<j<k) where all three pairwise distances ~ dmin.
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if (abs(D[i, j] - dmin) < 1e-8 and abs(D[j, k] - dmin) < 1e-8
                        and abs(D[i, k] - dmin) < 1e-8):
                    faces.append((i, j, k))
    faces_vec = []
    for (i, j, k) in faces:
        c = (verts[i] + verts[j] + verts[k]) / 3.0
        c = c / np.linalg.norm(c)
        faces_vec.append(c)
    faces_vec = np.array(faces_vec)

    return verts, edges_vec, faces_vec


def _lj_energy(pos):
    """Total Lennard-Jones energy 4*sum(r^-12 - r^-6) over pairs."""
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** -3
    return 4.0 * np.sum(r6 * r6 - r6)


def _lj_energy_and_grad(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** -3
    r12 = r6 * r6
    # dE/dr2 for one pair = 4*(-6*r2^-7*... ) careful: E_pair = 4*(r2^-6 - r2^-3)
    # dE/dr2 = 4*(-6*r2^-7 + 3*r2^-4)
    dE_dr2 = 4.0 * (-6.0 * r2 ** -7 + 3.0 * r2 ** -4)
    # grad over coordinates: for pair (i,j), dE/dpos_i = -2 * dE_dr2 * (pos_i - pos_j)
    grad = -2.0 * np.einsum('ij,ijk->ik', dE_dr2, diff)
    # note: off-diagonal counted once in the sum because dE_dr2 matrix is full and diff antisymmetric,
    # but we double count? Actually sum over all i,j (i!=j) of dE_dr2_ij * d(pos_i-pos_j)/dpos_i
    # The standard trick: grad = -2 * (dE_dr2 * diff) summed over j. This counts each pair twice
    # (once for i, once for j) which is correct because E = sum_{i<j} but using full matrix.
    # Let's verify: E = 0.5 * sum_{i!=j} 4*(r2^-6 - r2^-3)? No, we used full sum without 0.5 in energy.
    # In _lj_energy we summed over all i,j (i!=j) r6*r6 - r6, which is 2x the pair sum.
    # So energy there is 2x the true pair energy! Need to be consistent.
    # Let's redefine: true E = 4 * sum_{i<j} (r^-12 - r^-6). My _lj_energy sums over all i!=j,
    # giving 2 * true E. So I must halve. Let me fix _lj_energy to use 0.5 factor.
    return 0.5 * np.sum(4.0 * (r12 - r6)), grad


def _lj_energy_true(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** -3
    r12 = r6 * r6
    return 0.5 * np.sum(4.0 * (r12 - r6))


def _lj_grad(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    dE_dr2 = 0.5 * 4.0 * (-6.0 * r2 ** -7 + 3.0 * r2 ** -4)
    grad = -2.0 * np.einsum('ij,ijk->ik', dE_dr2, diff)
    return grad


def _radial_energy(params):
    """params = (r1, r2) radii for 1st and 2nd shells; center at origin.
    Returns total LJ energy of the 55-atom structure."""
    verts, edges_vec, faces_vec = _RADIAL_CACHE
    r1, r2 = params
    pos = np.zeros((55, 3))
    pos[1:13] = verts * r1
    pos[13:43] = edges_vec * r2
    pos[43:55] = faces_vec * r2
    return _lj_energy_true(pos)


_RADIAL_CACHE = _build_icosahedron()


def _make_positions(params):
    verts, edges_vec, faces_vec = _RADIAL_CACHE
    r1, r2 = params
    pos = np.zeros((55, 3))
    pos[1:13] = verts * r1
    pos[13:43] = edges_vec * r2
    pos[43:55] = faces_vec * r2
    return pos


def construct():
    t0 = time.time()
    deadline = 150.0 - 10.0  # safety margin

    # 1) Optimize the two shell radii with Nelder-Mead (low-dim, robust).
    # Start near known Mackay geometry: r1 ~ 1.12, r2 ~ 1.66 (approx).
    best_params = np.array([1.12, 1.66])
    best_e = _radial_energy(best_params)

    try:
        res = minimize(_radial_energy, best_params, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_e:
            best_e = res.fun
            best_params = res.x
    except Exception:
        pass

    if time.time() > deadline:
        return _make_positions(best_params)

    # 2) Full Cartesian L-BFGS polish from the symmetric structure.
    pos0 = _make_positions(best_params)
    try:
        res2 = minimize(_lj_energy_true, pos0, jac=_lj_grad, method='L-BFGS-B',
                        options={'maxiter': 4000, 'gtol': 1e-12, 'ftol': 1e-14})
        if res2.fun < best_e:
            best_e = res2.fun
            pos0 = res2.x
    except Exception:
        pass

    # 3) Optional second polish with even tighter tolerance if time remains.
    if time.time() < deadline - 20.0:
        try:
            res3 = minimize(_lj_energy_true, pos0, jac=_lj_grad, method='L-BFGS-B',
                            options={'maxiter': 4000, 'gtol': 1e-14, 'ftol': 1e-16})
            if res3.fun < best_e:
                best_e = res3.fun
                pos0 = res3.x
        except Exception:
            pass

    return pos0
# EVOLVE-BLOCK-END
