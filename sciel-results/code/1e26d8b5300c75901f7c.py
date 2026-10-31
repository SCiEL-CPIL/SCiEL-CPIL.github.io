"""Heilbronn triangles: 11 points in unit equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of unit equilateral triangle

def bary_to_cart(b):
    """Convert barycentric coordinates (u, v) with u>=0, v>=0, u+v<=1 to Cartesian.
    Triangle vertices: A=(0,0), B=(1,0), C=(0.5, H).
    Point = u*A + v*B + (1-u-v)*C = v*B + (1-u-v)*C
    x = v + (1-u-v)*0.5 = 0.5 + 0.5*v - 0.5*u
    y = (1-u-v)*H
    """
    u = b[:, 0]
    v = b[:, 1]
    x = 0.5 + 0.5 * v - 0.5 * u
    y = H * (1.0 - u - v)
    return np.column_stack([x, y])

def cart_to_bary(p):
    """Convert Cartesian to barycentric (u, v) where u for A, v for B."""
    x = p[:, 0]
    y = p[:, 1]
    # y = H*(1-u-v) => u+v = 1 - y/H
    # x = 0.5 + 0.5*v - 0.5*u => 2x - 1 = v - u
    s = 1.0 - y / H  # u + v
    d = 2.0 * x - 1.0  # v - u
    u = 0.5 * (s - d)
    v = 0.5 * (s + d)
    return np.column_stack([u, v])

def all_triangle_areas_cart(pts):
    """Compute areas of all C(n,3) triangles from n points in Cartesian coords."""
    n = len(pts)
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    p1, p2, p3 = pts[i], pts[j], pts[k]
    areas = 0.5 * np.abs(np.cross(p2 - p1, p3 - p1))
    return areas

def min_triangle_area(pts):
    """Minimum area over all C(n,3) triangles. pts in Cartesian."""
    return np.min(all_triangle_areas_cart(pts))

def soft_min(areas, beta):
    """Soft minimum via log-sum-exp: -(1/beta)*log(sum(exp(-beta*a)))"""
    a_min = np.min(areas)
    # For numerical stability
    return a_min - np.log(np.sum(np.exp(-beta * (areas - a_min)))) / beta

def objective_bary(b, beta):
    """Objective: -soft_min of triangle areas. b in barycentric (n,2)."""
    # Project to feasible region
    b = np.maximum(b, 0.0)
    s = b[:, 0] + b[:, 1]
    over = s > 1.0
    if np.any(over):
        b[over] /= s[over, np.newaxis]
    pts = bary_to_cart(b)
    areas = all_triangle_areas_cart(pts)
    return -soft_min(areas, beta)

def gradient_bary(b, beta):
    """Numerical gradient of objective w.r.t. barycentric coords."""
    eps = 1e-6
    grad = np.zeros_like(b)
    f0 = objective_bary(b, beta)
    for i in range(len(b)):
        for j in range(2):
            bp = b.copy()
            bp[i, j] += eps
            grad[i, j] = (objective_bary(bp, beta) - f0) / eps
    return grad

def random_bary_init(rng, n=N):
    """Generate random barycentric coordinates inside the triangle."""
    b = np.zeros((n, 2))
    for i in range(n):
        u = rng.random()
        v = rng.random()
        while u + v > 1.0:
            u = rng.random()
            v = rng.random()
        # Add small margin
        u = 0.02 + 0.96 * u
        v = 0.02 + 0.96 * v
        while u + v > 0.96:
            u = 0.02 + 0.96 * rng.random()
            v = 0.02 + 0.96 * rng.random()
        b[i] = [u, v]
    return b

def optimize_single(b0, beta, max_iter=200, time_limit=10.0):
    """Optimize from initial barycentric coords using L-BFGS-B with bounds."""
    t_start = time.time()
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N
    
    def obj(b):
        return objective_bary(b, beta)
    
    def grad(b):
        return gradient_bary(b, beta)
    
    result = minimize(
        obj, b0, method='L-BFGS-B', jac=grad,
        bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.x, result.fun

def polish(b, max_iter=500):
    """High-precision polish using higher beta."""
    t_start = time.time()
    beta = 200.0
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N
    
    def obj(b):
        return objective_bary(b, beta)
    
    def grad(b):
        return gradient_bary(b, beta)
    
    result = minimize(
        obj, b, method='L-BFGS-B', jac=grad,
        bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    return result.x

def construct():
    import random
    rng = random.Random(42)
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    best_bary = None
    best_score = -np.inf
    
    # Multi-restart with escalating beta
    betas = [5.0, 20.0, 50.0, 100.0, 200.0]
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        b0 = random_bary_init(rng)
        
        for beta in betas:
            if time.time() > deadline - 15.0:
                break
            
            try:
                b_opt, _ = optimize_single(b0, beta, max_iter=150, time_limit=15.0)
                b0 = b_opt
            except Exception:
                continue
        
        if time.time() > deadline - 10.0:
            break
        
        # Polish with high beta
        try:
            b_polished = polish(b0, max_iter=300)
        except Exception:
            b_polished = b0
        
        pts = bary_to_cart(b_polished)
        score = min_triangle_area(pts)
        
        if score > best_score:
            best_score = score
            best_bary = b_polished.copy()
    
    if best_bary is None:
        # Fallback: simple random
        b0 = random_bary_init(rng)
        best_bary = b0
    
    pts = bary_to_cart(best_bary)
    
    # Verify points are inside triangle
    u = best_bary[:, 0]
    v = best_bary[:, 1]
    mask = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
    if not np.all(mask):
        # Clamp
        best_bary = np.maximum(best_bary, 0.0)
        s = best_bary[:, 0] + best_bary[:, 1]
        over = s > 1.0
        if np.any(over):
            best_bary[over] /= s[over, np.newaxis]
        pts = bary_to_cart(best_bary)
    
    return pts
