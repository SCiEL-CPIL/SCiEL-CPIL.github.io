"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: evaluator kills after 150s, return best before 140s for safety
DEADLINE = 140.0
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def lj_energy(pos):
    """Compute total Lennard-Jones energy for a set of positions."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    # Compute all pair distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    # Avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # Sum over all pairs (each pair counted twice, so divide by 2)
    e = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(e, 0.0)
    return 0.5 * e.sum()


def lj_forces(pos):
    """Compute forces (negative gradient) for Lennard-Jones potential."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # Force on each atom: F_i = sum_j 4*(12*inv_r12 - 6*inv_r6) * (pos_i - pos_j) / r2
    # The factor (pos_i - pos_j)/r2 gives the unit vector direction
    coeff = 4.0 * (12.0 * inv_r12 - 6.0 * inv_r6)
    np.fill_diagonal(coeff, 0.0)
    forces = np.einsum('ij,ijk->ik', coeff, diff)
    return forces


def relax(positions, max_iter=200, tol=1e-10):
    """Local relaxation using L-BFGS-B."""
    def fun(x):
        pos = x.reshape(-1, 3)
        return lj_energy(pos)

    def jac(x):
        pos = x.reshape(-1, 3)
        return -lj_forces(pos).ravel()

    x0 = positions.ravel()
    res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50})
    return res.x.reshape(-1, 3)


def cubic_seed():
    """Generate a cubic lattice fragment seed."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12


def construct():
    global START_TIME
    START_TIME = time.time()

    rng = np.random.default_rng(42)

    # Start from cubic seed
    best_pos = cubic_seed()
    best_energy = lj_energy(best_pos)

    # First, relax the seed
    try:
        relaxed = relax(best_pos, max_iter=300, tol=1e-12)
        e = lj_energy(relaxed)
        if e < best_energy:
            best_pos = relaxed
            best_energy = e
    except Exception:
        pass

    # Basin hopping with L-BFGS relaxation
    # Use progressively larger perturbations to escape icosahedral funnel
    n_attempts = 0
    max_attempts = 200

    while _time_left() > 5.0 and n_attempts < max_attempts:
        n_attempts += 1

        # Perturbation size: start moderate, occasionally large to jump barriers
        # Icosahedral barrier is significant; use scale that can cross it
        if n_attempts % 10 == 0:
            # Occasional large jump
            scale = 1.5 + rng.random() * 1.0
        else:
            # Moderate perturbation
            scale = 0.3 + 0.5 * (1.0 - n_attempts / max_attempts) + rng.random() * 0.3

        # Generate perturbed candidate
        candidate = best_pos + scale * rng.standard_normal((N, 3))

        # Ensure minimum separation of 0.5
        # Simple check: if any pair is too close, nudge apart
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(candidate[i] - candidate[j])
                if d < 0.5:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            continue

        # Relax the candidate
        try:
            relaxed_cand = relax(candidate, max_iter=200, tol=1e-10)
            e = lj_energy(relaxed_cand)
            if e < best_energy:
                best_pos = relaxed_cand
                best_energy = e
                # After finding improvement, do a tighter polish
                if _time_left() > 10.0:
                    try:
                        polished = relax(best_pos, max_iter=500, tol=1e-13)
                        ep = lj_energy(polished)
                        if ep < best_energy:
                            best_pos = polished
                            best_energy = ep
                    except Exception:
                        pass
        except Exception:
            continue

    # Final high-precision polish pass
    if _time_left() > 3.0:
        try:
            polished = relax(best_pos, max_iter=1000, tol=1e-14)
            e = lj_energy(polished)
            if e < best_energy:
                best_pos = polished
                best_energy = e
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
