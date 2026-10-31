"""
LJ38 cluster global minimum search.

Strategy: Generate the 38-atom FCC truncated-octahedron by taking the 38
closest points in the FCC lattice to the origin, then polish with L-BFGS.
This directly targets the known global minimum geometry.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _generate_fcc_truncated_octahedron():
    """
    Generate the 38-atom FCC truncated-octahedron by taking the 38 closest
    FCC lattice points to the origin.
    
    FCC lattice points: (i, j, k) where i+j+k is even.
    """
    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i * i + j * j + k * k
                    pts.append((float(i), float(j), float(k), d2))
    
    pts = np.array(pts)
    # Sort by distance from origin
    order = np.argsort(pts[:, 3])
    pts = pts[order]
    
    # Take the 38 closest
    selected = pts[:N, :3]
    return selected

def _relax(pos, max_iter=500, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure minimum inter-atomic separation."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill

    # Generate the 38-atom FCC truncated octahedron
    pos = _generate_fcc_truncated_octahedron()
    
    # Center the structure
    pos -= np.mean(pos, axis=0)
    
    # Scale so that nearest-neighbor distance is 2^(1/6) (LJ minimum)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pos = pos * scale
    
    # Relax
    best_pos = pos.copy()
    best_energy, _ = _lj_energy_and_grad(best_pos)
    
    try:
        p, e = _relax(pos, max_iter=500, gtol=1e-12)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        pass
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        try:
            p, e = _relax(best_pos, max_iter=1000, gtol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass
    
    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
