"""Rewritten program: continuous relaxation on a high-res geodesic grid + gradient descent + local polish."""
import time
import math
import numpy as np
from scipy.spatial import cKDTree

N = 100
GRID_N = 200  # ~200*198 ≈ 39600 candidate vertices (enough resolution)

# Pre-compute geodesic grid vertices (Fibonacci spiral at high resolution)
def _make_grid(M):
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

_GRID = _make_grid(GRID_N)
_GRID_N = len(_GRID)

def _energy_fsum(pts):
    """Compute log(1/d) energy using math.fsum for precision."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad(pts):
    """
    Compute energy and gradient for points on sphere.
    Energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    Gradient w.r.t. xi: sum_{j!=i} (xi - xj) / |xi - xj|^2  (for -log dist)
    Actually: d/dxi [-log|xi-xj|] = -(xi-xj)/|xi-xj|^2
    So gradient of energy = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    """
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_vec = pts[i] - pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-12:
                return float('inf'), grad
            energy -= 0.5 * math.log(d)  # each pair counted twice, so halve
            # gradient contribution: -d/dxi log|xi-xj| = -(xi-xj)/|xi-xj|^2
            grad[i] += -d_vec / (d * d)
    # energy was halved per pair but we summed over all j!=i, so each pair appears twice
    # Let me recompute properly:
    energy = 0.0
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = pts[i] - pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-12:
                return float('inf'), grad
            energy -= math.log(d)
            # d/dxi (-log d) = -(xi-xj)/d^2
            gi = -d_vec / (d * d)
            # d/dxj (-log d) = -xj/|xi-xj|^2 ... wait
            # log|xi-xj|, d/dxi = (xi-xj)/|xi-xj|^2, so -log gives -(xi-xj)/|xi-xj|^2
            # d/dxj = -(xj-xi)/|xi-xj|^2 = (xi-xj)/|xi-xj|^2
            gj = d_vec / (d * d)
            grad[i] += gi
            grad[j] += gj
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # 5s safety margin

    def time_left():
        return DEADLINE - (time.time() - t0)

    # Start with Fibonacci spiral
    pts = _make_grid(N)
    best_pts = pts.copy()
    best_energy = _energy_fsum(best_pts)

    # Phase 1: Gradient descent on the sphere using Riemannian-like steps
    lr = 0.01
    for step in range(20000):
        if time_left() < 10.0:
            break
        energy, grad = _energy_and_grad(pts)
        if not math.isfinite(energy):
            break
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        # Project gradient to tangent space
        pts_tan = pts.copy()
        grad_tan = np.zeros_like(grad)
        for i in range(N):
            proj = np.dot(grad[i], pts[i])
            grad_tan[i] = grad[i] - proj * pts[i]
        # Gradient descent step
        pts = pts - lr * grad_tan
        pts = _project_to_sphere(pts)
        # Adapt learning rate
        if step % 100 == 0:
            lr = max(0.001, lr * 0.95)

    # Phase 2: Local polish with smaller steps
    lr = 0.001
    for step in range(20000):
        if time_left() < 5.0:
            break
        energy, grad = _energy_and_grad(pts)
        if not math.isfinite(energy):
            break
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        grad_tan = np.zeros_like(grad)
        for i in range(N):
            proj = np.dot(grad[i], pts[i])
            grad_tan[i] = grad[i] - proj * pts[i]
        pts = pts - lr * grad_tan
        pts = _project_to_sphere(pts)
        if step % 200 == 0:
            lr = max(0.0001, lr * 0.9)

    # Phase 3: Coordinate descent polish
    for sweep in range(200):
        if time_left() < 3.0:
            break
        improved = False
        for i in range(N):
            # Optimize point i by trying small perturbations in tangent plane
            p = pts[i]
            # Find tangent basis
            if abs(p[2]) < 0.9:
                e1 = np.array([-p[1], p[0], 0.0])
            else:
                e1 = np.array([0.0, -p[2], p[1]])
            e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(p, e1)
            e2 = e2 / np.linalg.norm(e2)
            
            best_p = p
            best_e = energy
            for angle in np.linspace(0, 2*np.pi, 36):
                for rad in [0.001, 0.005, 0.01, 0.02]:
                    new_p = p + rad * (np.cos(angle) * e1 + np.sin(angle) * e2)
                    new_p = new_p / np.linalg.norm(new_p)
                    pts[i] = new_p
                    e = _energy_fsum(pts)
                    if e < best_e:
                        best_e = e
                        best_p = new_p.copy()
            pts[i] = best_p
            if not math.isfinite(best_e):
                break
        if math.isfinite(best_e) and best_e < best_energy:
            best_energy = best_e
            best_pts = pts.copy()

    # Final validation
    result = _project_to_sphere(best_pts)
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(result[i] - result[j]) < 1e-9:
                # Nudge apart
                result[j] += 1e-8 * np.array([1, 0, 0])
                result[j] /= np.linalg.norm(result[j])
    
    return result
