"""
Structural change: smooth log-sum-exp objective with icosahedron-based initialization
and Nelder-Mead local search. Avoids numerical gradient computation by using
a smooth surrogate that has good gradient behavior for L-BFGS-B.
"""
import time
import numpy as np
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

_DEADLINE = 140.0
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def _pairwise_sq_dists(P):
    """Return squared pairwise distances (upper triangle flat)."""
    diff = P[:, None, :] - P[None, :, :]
    sq = np.einsum('ijk,ijk->ij', diff, diff)
    iu = np.triu_indices(N, k=1)
    return sq[iu]


def _raw_score(P):
    """Compute (max/min)^2 ratio of pairwise distances."""
    d2 = _pairwise_sq_dists(P)
    mn = d2.min()
    mx = d2.max()
    if mn < 1e-12:
        return np.inf
    return (mx / mn) ** 2


def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: log-sum-exp based objective.
    We want to minimize the ratio max_d2 / min_d2.
    Using log-sum-exp:
      log(max_d2) ≈ (1/beta) * log(sum(exp(beta * d2 / max_d2)))
    But a simpler approach: minimize (log_sum_exp(d2) - log_sum_exp(-d2))
    which approximates (max_d2 - min_d2) in a smooth way, scaled.
    
    Actually, let's use: f = log(sum(exp(beta*d2))) - log(sum(exp(-beta*d2)))
    This is smooth and its minimum corresponds to making all d2 equal.
    """
    P = x.reshape(N, D)
    d2 = _pairwise_sq_dists(P)
    if d2.min() < 1e-10:
        return 1e10
    
    # Normalize to avoid overflow
    d2_shifted = d2 - d2.mean()
    
    # log-sum-exp for positive and negative
    max_pos = np.max(d2_shifted)
    lse_pos = max_pos + np.log(np.sum(np.exp(beta * d2_shifted - beta * max_pos)))
    
    max_neg = np.max(-d2_shifted)
    lse_neg = max_neg + np.log(np.sum(np.exp(-beta * d2_shifted + beta * max_neg)))
    
    return (lse_pos - lse_neg) / beta


def _gradient_smooth(x, beta=50.0):
    """
    Analytical gradient of the smooth objective.
    f = (1/beta) * [log(sum(exp(beta*d2_i))) - log(sum(exp(-beta*d2_i)))]
    df/dx = (1/beta) * [beta * sum_i exp(beta*d2_i)/sum_j exp(beta*d2_j) * dd2_i/dx
                        - beta * sum_i exp(-beta*d2_i)/sum_j exp(-beta*d2_j) * (-dd2_i/dx)]
          = sum_i p_pos_i * dd2_i/dx + sum_i p_neg_i * dd2_i/dx
    where p_pos_i = exp(beta*d2_i)/sum(exp(beta*d2_j)), p_neg_i = exp(-beta*d2_i)/sum(exp(-beta*d2_j))
    
    dd2_i/dx_k = 2*(P_i - P_j) for pair (i,j) corresponding to d2_k
    """
    P = x.reshape(N, D)
    d2 = _pairwise_sq_dists(P)
    if d2.min() < 1e-10:
        return np.zeros_like(x)
    
    d2_shifted = d2 - d2.mean()
    
    max_pos = np.max(d2_shifted)
    exp_pos = np.exp(beta * d2_shifted - beta * max_pos)
    sum_pos = np.sum(exp_pos)
    p_pos = exp_pos / sum_pos
    
    max_neg = np.max(-d2_shifted)
    exp_neg = np.exp(-beta * d2_shifted + beta * max_neg)
    sum_neg = np.sum(exp_neg)
    p_neg = exp_neg / sum_neg
    
    # Combined weight: p_pos + p_neg for each pair
    weights = p_pos + p_neg
    
    # Now compute gradient: sum over pairs of weights[i] * 2*(P_a - P_b)
    # We need the pair indices
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    grad = np.zeros_like(x)
    for k in range(NUM_PAIRS):
        w = weights[k]
        if w < 1e-15:
            continue
        a, b = i_idx[k], j_idx[k]
        diff = 2.0 * (P[a] - P[b])
        grad[a * D:(a + 1) * D] += w * diff
        grad[b * D:(b + 1) * D] -= w * diff
    
    return grad


def _generate_icosahedron():
    """
    Generate 12 vertices of a regular icosahedron, plus 2 additional points
    to make 14. The icosahedron has 12 vertices, 30 edges, 20 faces.
    We'll add 2 points at the center of opposite faces or along an axis.
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1],
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Add 2 more points: opposite poles or face centers
    # Let's add points at (0,0,1) and (0,0,-1) direction but projected
    # Actually, let's add two points that are roughly equidistant
    extra1 = np.array([0.0, 0.0, 1.0])
    extra2 = np.array([0.0, 0.0, -1.0])
    
    # Scale to be at similar radius as icosahedron vertices
    ico_norm = np.linalg.norm(verts[0])
    extra1 = extra1 * ico_norm
    extra2 = extra2 * ico_norm
    
    P = np.vstack([verts, extra1[None], extra2[None]])
    return P


def _generate_fib_sphere(n):
    """Generate n points on a sphere using Fibonacci lattice."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = indices / n - 1 + 1.0 / n
    r = np.sqrt(1 - z * z)
    theta = phi * indices
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    P = np.stack([x, y, z], axis=1)
    return P


def _generate_tetrahedron_midpoints():
    """Tetrahedron vertices + edge midpoints + face centroids = 14 points."""
    a = 1.0 / np.sqrt(2)
    verts = np.array([
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a],
    ])
    verts *= np.sqrt(2)
    
    points = [verts]
    for i in range(4):
        for j in range(i + 1, 4):
            mid = (verts[i] + verts[j]) / 2.0
            points.append(mid)
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for f in faces:
        centroid = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        points.append(centroid)
    
    P = np.vstack(points)
    return P


def _polish_lbfgs(P_init, beta=50.0, maxiter=300):
    """Polish using L-BFGS-B with analytical gradient."""
    x0 = P_init.flatten()
    
    def obj(x):
        return _smooth_objective(x, beta)
    
    def grad(x):
        return _gradient_smooth(x, beta)
    
    result = minimize(
        obj, x0, method='L-BFGS-B', jac=grad,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    return result.x.reshape(N, D)


def _polish_nelder_mead(P_init, maxiter=500):
    """Polish using Nelder-Mead with raw score."""
    x0 = P_init.flatten()
    
    def obj(x):
        P = x.reshape(N, D)
        return _raw_score(P)
    
    result = minimize(
        obj, x0, method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
    )
    return result.x.reshape(N, D)


def _generate_initial_guesses():
    """Generate diverse initial guesses."""
    guesses = []
    
    # Icosahedron-based
    P_ico = _generate_icosahedron()
    guesses.append(P_ico)
    
    # Fibonacci sphere
    P_fib = _generate_fib_sphere(N)
    guesses.append(P_fib)
    
    # Tetrahedron + midpoints
    P_tet = _generate_tetrahedron_midpoints()
    guesses.append(P_tet)
    
    # Random spherical points with different seeds
    rng = np.random.RandomState(123)
    for seed in range(5):
        rng = np.random.RandomState(seed * 1000 + 42)
        P_rand = rng.randn(N, D)
        norms = np.linalg.norm(P_rand, axis=1, keepdims=True)
        P_rand = P_rand / norms
        # Add some radial variation
        radii = 1.0 + rng.randn(N, 1) * 0.2
        P_rand = P_rand * radii
        guesses.append(P_rand)
    
    return guesses


def construct():
    global _t0
    _t0 = time.time()
    
    best_P = None
    best_score = np.inf
    
    # Generate diverse initial guesses
    guesses = _generate_initial_guesses()
    
    for guess in guesses:
        if _elapsed() > _DEADLINE - 20:
            break
        
        # Polish with L-BFGS-B using smooth objective
        try:
            P_polished = _polish_lbfgs(guess, beta=50.0, maxiter=200)
            s = _raw_score(P_polished)
            if s < best_score:
                best_score = s
                best_P = P_polished.copy()
        except Exception:
            continue
        
        if _elapsed() > _DEADLINE - 15:
            break
    
    # Final high-precision polish with Nelder-Mead
    if best_P is not None and _elapsed() < _DEADLINE - 10:
        try:
            P_nm = _polish_nelder_mead(best_P, maxiter=300)
            s_nm = _raw_score(P_nm)
            if s_nm < best_score:
                best_score = s_nm
                best_P = P_nm
        except Exception:
            pass
    
    # One more L-BFGS-B polish with higher beta for precision
    if best_P is not None and _elapsed() < _DEADLINE - 5:
        try:
            P_final = _polish_lbfgs(best_P, beta=100.0, maxiter=100)
            s_final = _raw_score(P_final)
            if s_final < best_score:
                best_score = s_final
                best_P = P_final
        except Exception:
            pass
    
    # Verify distinctness
    if best_P is None:
        # Fallback: return icosahedron-based
        best_P = _generate_icosahedron()
    
    d2 = _pairwise_sq_dists(best_P)
    if d2.min() < 1e-10:
        rng = np.random.RandomState(0)
        best_P += rng.randn(N, D) * 1e-6
    
    return best_P


if __name__ == '__main__':
    P = construct()
    print("Score:", _raw_score(P))
