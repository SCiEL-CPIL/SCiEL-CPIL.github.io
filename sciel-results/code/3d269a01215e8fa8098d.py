"""Packing unit regular hexagons inside a regular hexagon."""
import time
import numpy as np

N = 11
DEADLINE = 150.0
MARGIN = 8.0


def _hex_circumradius(side):
    return float(side)


def _pairwise_dists(P):
    # P: (n,2) array; returns (n,n) distance matrix
    d2 = np.sum(P ** 2, axis=1)
    D2 = d2[:, None] + d2[None, :] - 2.0 * (P @ P.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2)


def _max_dist_from_center(P, c):
    Q = P - np.array(c)
    return float(np.max(np.sqrt(np.sum(Q ** 2, axis=1))))


def _feasible(P, c, L):
    # Check pairwise distances >= 2 and max distance from center <= L-1
    D = _pairwise_dists(P)
    if np.min(D[np.triu_indices_from(D, k=1)]) < 2.0 - 1e-9:
        return False
    if _max_dist_from_center(P, c) > (L - 1.0) + 1e-9:
        return False
    return True


def _hex_lattice_positions(n):
    """Generate n positions on a hexagonal lattice with spacing 2.0, centered at origin."""
    # Hex lattice: basis vectors at 60 deg
    a = 2.0
    v1 = np.array([a, 0.0])
    v2 = np.array([a * 0.5, a * np.sqrt(3.0) / 2.0])
    pts = []
    # Spiral outward from origin
    if n <= 1:
        pts.append(np.array([0.0, 0.0]))
        return np.array(pts)
    pts.append(np.array([0.0, 0.0]))
    ring = 1
    while len(pts) < n:
        # Ring r has 6*r points
        for edge in range(6):
            # Direction of edge
            theta = np.pi / 3.0 * edge
            # Start of edge
            if edge == 0:
                start = ring * v1
            elif edge == 1:
                start = ring * (v1 + v2) - v1  # corner
            # Simpler: generate all lattice points with hex distance == ring
            pass
        # Generate all points at hex distance == ring
        for i in range(-ring, ring + 1):
            for j in range(-ring, ring + 1):
                # Hex distance = max(|i|, |j|, |i+j|) in axial coords
                k = -i - j
                hd = max(abs(i), abs(j), abs(k))
                if hd == ring:
                    p = i * v1 + j * v2
                    pts.append(p)
        ring += 1
    return np.array(pts[:n])


def _optimize_positions(P0, c, L_target, max_iter=2000, tol=1e-8):
    """Local optimization to minimize max distance from center subject to pairwise >= 2."""
    from scipy.optimize import minimize

    n = len(P0)
    x0 = P0.flatten()

    def objective(x):
        P = x.reshape(n, 2)
        Q = P - np.array(c)
        dists = np.sqrt(np.sum(Q ** 2, axis=1))
        # Smooth max via log-sum-exp
        m = np.max(dists)
        # Add penalty for pairwise violations
        D = _pairwise_dists(P)
        lower = np.tril_indices_from(D, k=-1)
        pair_d = D[lower]
        # Penalty: sum of (2.0 - d)^2 for d < 2.0
        viol = np.maximum(2.0 - pair_d, 0.0)
        pen = 100.0 * np.sum(viol ** 2)
        return m + pen

    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': tol})
    return res.x.reshape(n, 2)


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN

    # Generate initial lattice positions
    P = _hex_lattice_positions(N)
    c = (0.0, 0.0)

    # Compute initial L
    max_d = _max_dist_from_center(P, c)
    L = max_d + 1.0

    best_L = L
    best_P = P.copy()

    # Check feasibility
    if _feasible(P, c, L):
        best_L = L
        best_P = P.copy()

    # Try local optimization
    if time.time() < deadline:
        # Optimize with a slightly tighter target to encourage compaction
        L_target = L * 0.98
        P_opt = _optimize_positions(P, c, L_target, max_iter=1000, tol=1e-8)
        max_d_opt = _max_dist_from_center(P_opt, c)
        L_opt = max_d_opt + 1.0

        if _feasible(P_opt, c, L_opt):
            if L_opt < best_L:
                best_L = L_opt
                best_P = P_opt.copy()

    # Try a second round of optimization from the optimized positions
    if time.time() < deadline:
        P_opt2 = _optimize_positions(best_P, c, best_L * 0.97, max_iter=1500, tol=1e-9)
        max_d_opt2 = _max_dist_from_center(P_opt2, c)
        L_opt2 = max_d_opt2 + 1.0

        if _feasible(P_opt2, c, L_opt2):
            if L_opt2 < best_L:
                best_L = L_opt2
                best_P = P_opt2.copy()

    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(best_P[i, 0]), float(best_P[i, 1]), 0.0])

    return inner, c, float(best_L), 0.0


# EVOLVE-BLOCK-START
# (construct is defined above; this block is kept for compatibility)
# EVOLVE-BLOCK-END
