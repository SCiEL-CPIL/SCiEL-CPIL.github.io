# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 620.0
_t0 = time.time()

def _remaining():
    return _DEADLINE - time.time()

# Precompute unit hexagon vertex offsets (side=1, angle=0)
_ANG0 = np.deg2rad(60.0 * np.arange(6))
_COS0 = np.cos(_ANG0)
_SIN0 = np.sin(_ANG0)
_VERTS0 = np.column_stack([_COS0, _SIN0])  # (6,2)

# Outer hexagon inward normals as function of phi (angle of outer)
def _outer_inward_normals(phi_deg):
    phi = np.deg2rad(phi_deg)
    # outward edge normals for outer hexagon at angle phi
    ang_out = phi + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    out_n = np.column_stack([np.cos(ang_out), np.sin(ang_out)])
    return -out_n  # inward normals

def _inner_supports(positions, angles_deg):
    """
    positions: (11,2), angles_deg: (11,)
    Returns (11,6): support function of each inner hexagon along the 6
    fixed directions of a unit hexagon (angle 0). This is rotation-invariant:
    support of a rotated hexagon along direction d equals support of base
    hexagon along R^T d.
    """
    # For each inner hex i with rotation a_i, support along direction d_k
    # (k=0..5, the 6 directions of a unit hex at angle 0) is:
    #   (position_i) . d_k + max_j ( (cos(a_i + 60j), sin(a_i + 60j)) . d_k )
    # The max term depends only on (a_i - 60k) mod 60, which is periodic.
    # Compute max over j of cos(a_i + 60j - 60k) = cos(a_i - 60k + 60j)
    # = max over j of cos((a_i - 60k) + 60j). This is the support of a unit
    # hexagon (angle 0) along direction at angle (a_i - 60k).
    # Actually simpler: support of rotated hex along d = max_j (R(a_i) v0_j) . d
    # = max_j v0_j . R(-a_i) d
    # Let d = (cos 60k, sin 60k). R(-a_i) d has angle 60k - a_i.
    # max_j v0_j . (cos(60k-a_i), sin(60k-a_i)) = support of unit hex along
    # direction at angle (60k - a_i).
    # The support of a unit regular hexagon (angle 0) along direction at
    # angle beta is: max_j cos(beta - 60j) for j=0..5.
    # This is a periodic function of beta with period 60 deg.
    # Precompute: for beta in [0,60), support = max_j cos(beta - 60j).
    # By symmetry, for beta in [0,30]: support = cos(beta) (since j=0 gives
    # cos(beta), and this is the max). For beta in [30,60]: support = cos(beta-60)
    # = cos(60-beta) by symmetry. Actually let's just compute numerically.

    # Vectorized: for each (i,k), beta = 60k - a_i (in degrees)
    # beta_mod = ((60*k - a_i) mod 60) in [0,60)
    # support = max_j cos(beta_mod - 60j) ... but by periodicity = f(beta_mod)
    # where f(b) = max_j cos(b - 60j) for b in [0,60).
    # f(b) for b in [0,30] = cos(b) (j=0 wins)
    # f(b) for b in [30,60] = cos(b-60) = cos(60-b) (j=1 wins, by symmetry)
    # Actually f is even about 30: f(b) = cos(min(b, 60-b)) for b in [0,60].
    # Let's verify: b=0: cos(0)=1, max_j cos(-60j)=cos(0)=1. OK.
    # b=30: cos(30)=0.866, max_j cos(30-60j): j=0:cos30=0.866, j=1:cos(-30)=0.866. OK.
    # b=15: cos(15)=0.966, j=0:cos15=0.966, j=1:cos(-45)=0.707. max=0.966. OK.

    k_idx = np.arange(6)  # 0..5
    # beta in degrees: 60*k - a_i
    beta = 60.0 * k_idx[None, :] - angles_deg[:, None]  # (11, 6)
    beta_mod = np.mod(beta, 60.0)  # [0, 60)
    # f(b) = cos(min(b, 60-b))
    b_eff = np.minimum(beta_mod, 60.0 - beta_mod)  # [0, 30]
    support_rot = np.cos(np.deg2rad(b_eff))  # (11, 6)
    # Add position dot d_k
    d_k = np.column_stack([np.cos(np.deg2rad(60.0 * k_idx)),
                            np.sin(np.deg2rad(60.0 * k_idx))])  # (6,2)
    pos_dot = positions @ d_k.T  # (11, 6)
    return pos_dot + support_rot  # (11, 6)

def _min_outer_side(positions, angles_deg):
    """
    Find minimum S and best outer angle phi such that all inner hexagons fit
    in outer hexagon (center 0, side S, angle phi).
    """
    supports = _inner_supports(positions, angles_deg)  # (11, 6)

    def _required_S(phi_deg):
        in_n = _outer_inward_normals(phi_deg)  # (6, 2)
        # For each inner hex i, for each outer edge m:
        #   in_n[m] . (positions[i] + rotated_verts) <= S * sqrt(3)/2
        # The max over inner vertices of in_n[m] . vertex =
        #   in_n[m] . positions[i] + max_j (in_n[m] . R(a_i) v0_j)
        # = in_n[m] . positions[i] + support of unit hex (angle a_i) along in_n[m]
        # = in_n[m] . positions[i] + f(angle(in_n[m]) - a_i mod 60)
        # where f is the same support function.
        # So we need for all i, m:
        #   in_n[m] . positions[i] + f(angle(in_n[m]) - a_i) <= S * sqrt(3)/2
        # S >= max_{i,m} [in_n[m] . positions[i] + f(...)] / (sqrt(3)/2)

        in_n_angles = np.arctan2(in_n[:, 1], in_n[:, 0]) * 180.0 / np.pi  # (6,)
        # For each (i, m): beta = angle(in_n[m]) - a_i
        beta = in_n_angles[None, :] - angles_deg[:, None]  # (11, 6)
        beta_mod = np.mod(beta, 60.0)
        b_eff = np.minimum(beta_mod, 60.0 - beta_mod)
        supp = np.cos(np.deg2rad(b_eff))  # (11, 6)
        pos_dot = positions @ in_n.T  # (11, 6)
        max_val = np.max(pos_dot + supp)
        return max_val / (np.sqrt(3.0) / 2.0)

    # Try a grid of outer angles, find best
    best_S = np.inf
    best_phi = 0.0
    for phi in np.arange(0.0, 30.0, 1.0):  # 30 unique by symmetry
        S = _required_S(phi)
        if S < best_S:
            best_S = S
            best_phi = phi
    # Refine around best_phi
    for phi in np.arange(best_phi - 3.0, best_phi + 3.0, 0.25):
        S = _required_S(phi)
        if S < best_S:
            best_S = S
            best_phi = phi
    for phi in np.arange(best_phi - 0.5, best_phi + 0.5, 0.05):
        S = _required_S(phi)
        if S < best_S:
            best_S = S
            best_phi = phi
    return best_S, best_phi

def _separation_violation(positions):
    """
    Return max violation of separation constraint. Two hexagons i,j overlap
    iff ||pos_i - pos_j|| <= 2.0. Violation = 2.0 - dist if dist < 2.0, else 0.
    Returns max violation (0 if all separated).
    """
    # (11, 11) distance matrix
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist > 2.0 + 1e-6:
        return 0.0
    return max(0.0, 2.0 - min_dist)

def _objective(params, deadline):
    if time.time() > deadline:
        return 1e10
    positions = params[:22].reshape(11, 2)
    angles = params[22:]
    # Separation penalty
    viol = _separation_violation(positions)
    if viol > 0:
        return 1e8 * viol + _min_outer_side(positions, angles)[0]
    S, phi = _min_outer_side(positions, angles)
    return S

def _seed_positions():
    """Classic 11-hexagon packing: center + 6 around + 4 more."""
    # Hex grid: center at origin, neighbors at distance 2
    # 6 neighbors at angles 0, 60, 120, 180, 240, 300
    pos = np.zeros((11, 2))
    pos[0] = [0, 0]
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        pos[k + 1] = [2.0 * np.cos(ang), 2.0 * np.sin(ang)]
    # 4 more: at the "gaps" - actually for 11 hexagons, we need a tighter pack.
    # Let's use: center, 6 ring, and 4 at distance ~2 in the gaps between ring hexes.
    # Actually the optimal 11-hex packing in a hexagon: let me use a known good seed.
    # Place 4 additional hexagons at the concave vertices of the 7-hex cluster.
    # The 7-hex cluster has "notches" at angles 30, 90, 150, 210, 270, 330 at distance ~2*sqrt(3)/2*2 = ...
    # Let's just use a reasonable seed and let optimization fix it.
    for k in range(4):
        ang = np.deg2rad(60.0 * k + 30.0)
        pos[k + 7] = [2.5 * np.cos(ang), 2.5 * np.sin(ang)]
    return pos

def hexagon_packing_11():
    global _t0
    _t0 = time.time()
    deadline = _t0 + 610.0  # safety margin

    seed_pos = _seed_positions()
    seed_angles = np.zeros(11)
    x0 = np.concatenate([seed_pos.flatten(), seed_angles])

    # Evaluate seed
    best_S, best_phi = _min_outer_side(seed_pos, seed_angles)
    best_params = x0.copy()

    # Multi-start Nelder-Mead
    rng = np.random.default_rng(42)
    starts = [x0.copy()]
    # Add perturbed starts
    for _ in range(8):
        p = seed_pos + 0.3 * rng.standard_normal((11, 2))
        a = 10.0 * rng.standard_normal(11)
        starts.append(np.concatenate([p.flatten(), a]))

    for start in starts:
        if time.time() > deadline - 10:
            break
        # Quick check: is this start feasible (separated)?
        pos_check = start[:22].reshape(11, 2)
        if _separation_violation(pos_check) > 0.5:
            continue
        try:
            res = minimize(
                _objective,
                start,
                args=(deadline - 10,),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7,
                          'adaptive': True, 'maxfev': 2000}
            )
            if res.fun < best_S and res.fun < 1e7:
                # Verify separation
                pos_r = res.x[:22].reshape(11, 2)
                ang_r = res.x[22:]
                if _separation_violation(pos_r) <= 1e-6:
                    S_r, phi_r = _min_outer_side(pos_r, ang_r)
                    if S_r < best_S:
                        best_S = S_r
                        best_params = res.x.copy()
                        best_phi = phi_r
        except Exception:
            pass

    # Final refinement: local optimization around best
    if time.time() < deadline - 15:
        try:
            res = minimize(
                _objective,
                best_params,
                args=(deadline - 5,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                          'adaptive': True, 'maxfev': 3000}
            )
            if res.fun < best_S and res.fun < 1e7:
                pos_r = res.x[:22].reshape(11, 2)
                ang_r = res.x[22:]
                if _separation_violation(pos_r) <= 1e-6:
                    S_r, phi_r = _min_outer_side(pos_r, ang_r)
                    if S_r < best_S:
                        best_S = S_r
                        best_params = res.x.copy()
                        best_phi = phi_r
        except Exception:
            pass

    positions = best_params[:22].reshape(11, 2)
    angles = best_params[22:]

    # Final S and phi
    final_S, final_phi = _min_outer_side(positions, angles)

    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0.0, 0.0, final_phi])
    outer_hex_side_length = float(final_S)

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
