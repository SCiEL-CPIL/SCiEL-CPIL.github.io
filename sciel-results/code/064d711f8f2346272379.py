"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# Vertices of equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

# Precompute all pairs of indices for triangles
import itertools
tri_indices = list(itertools.combinations(range(N), 3))
n_tris = len(tri_indices)


def bary_to_cart(bary):
    """Convert barycentric coords (u, v) for each point to cartesian."""
    # bary shape: (N, 2), point = u*B + v*C + (1-u-v)*A = u*B + v*C
    # Since A is origin: point = u*B + v*C
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * B[0] + v * C[0]
    y = u * B[1] + v * C[1]
    return np.column_stack([x, y])


def min_triangle_area(points):
    """Compute the minimum triangle area over all triples."""
    areas = np.empty(n_tris)
    for i, (a, b, c) in enumerate(tri_indices):
        pa, pb, pc = points[a], points[b], points[c]
        # Area = 0.5 * |cross(pb - pa, pc - pa)|
        cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        areas[i] = 0.5 * abs(cross)
    return np.min(areas)


def soft_min_objective(bary, tau=0.01):
    """Smooth surrogate for -min_triangle_area using log-sum-exp."""
    points = bary_to_cart(bary)
    areas = np.empty(n_tris)
    for i, (a, b, c) in enumerate(tri_indices):
        pa, pb, pc = points[a], points[b], points[c]
        cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        areas[i] = 0.5 * abs(cross)
    # Soft min: -logsumexp(-areas/tau) * tau approximates min(areas)
    # We want to maximize min, so minimize -soft_min
    logsum = logsumexp(-areas / tau)
    soft_min_val = -logsum * tau
    return -soft_min_val


def soft_min_grad(bary, tau=0.01):
    """Gradient of soft_min_objective."""
    points = bary_to_cart(bary)
    areas = np.empty(n_tris)
    for i, (a, b, c) in enumerate(tri_indices):
        pa, pb, pc = points[a], points[b], points[c]
        cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        areas[i] = 0.5 * abs(cross)
    
    # p_i = exp(-areas_i/tau) / sum(exp(-areas_j/tau))
    neg_areas_over_tau = -areas / tau
    logsum = logsumexp(neg_areas_over_tau)
    p = np.exp(neg_areas_over_tau - logsum)  # shape (n_tris,)
    
    # d(soft_min)/d(area_i) = -p_i
    # soft_min = -tau * logsumexp(-areas/tau)
    # d(soft_min)/d(areas_i) = -tau * (-1/tau) * p_i = p_i
    # So d(-soft_min)/d(areas_i) = -p_i
    
    # Now d(areas_i)/d(point_j) for each point j in triangle i
    # area_i = 0.5 * |cross(pb - pa, pc - pa)|
    # Let's compute gradient w.r.t. each point
    
    grad_bary = np.zeros_like(bary)  # (N, 2)
    
    for i, (a, b, c) in enumerate(tri_indices):
        pa, pb, pc = points[a], points[b], points[c]
        # cross = (pb[0]-pa[0])*(pc[1]-pa[1]) - (pb[1]-pa[1])*(pc[0]-pa[0])
        # Let dx1 = pb[0]-pa[0], dy1 = pb[1]-pa[1], dx2 = pc[0]-pa[0], dy2 = pc[1]-pa[1]
        # cross = dx1*dy2 - dy1*dx2
        # d(cross)/d(pa) = (-dy2, dx2)
        # d(cross)/d(pb) = (dy2, -dx2)
        # d(cross)/d(pc) = (-dy1, dx1)
        
        dx1 = pb[0] - pa[0]
        dy1 = pb[1] - pa[1]
        dx2 = pc[0] - pa[0]
        dy2 = pc[1] - pa[1]
        cross = dx1 * dy2 - dy1 * dx2
        
        if abs(cross) < 1e-12:
            continue
        
        sign = 1.0 if cross > 0 else -1.0
        
        # d(area)/d(pa) = 0.5 * sign * (-dy2, dx2)
        # d(area)/d(pb) = 0.5 * sign * (dy2, -dx2)
        # d(area)/d(pc) = 0.5 * sign * (-dy1, dx1)
        
        darea_dpa = 0.5 * sign * np.array([-dy2, dx2])
        darea_dpb = 0.5 * sign * np.array([dy2, -dx2])
        darea_dpc = 0.5 * sign * np.array([-dy1, dx1])
        
        coeff = -p[i]  # d(-soft_min)/d(area_i)
        
        # d(-soft_min)/d(point_a) = coeff * darea_dpa
        # point = u*B + v*C, so d(point)/d(u) = B, d(point)/d(v) = C
        # d(-soft_min)/d(u_a) = dot(d(-soft_min)/d(point_a), B)
        # d(-soft_min)/d(v_a) = dot(d(-soft_min)/d(point_a), C)
        
        grad_bary[a, 0] += coeff * np.dot(darea_dpa, B)
        grad_bary[a, 1] += coeff * np.dot(darea_dpa, C)
        grad_bary[b, 0] += coeff * np.dot(darea_dpb, B)
        grad_bary[b, 1] += coeff * np.dot(darea_dpb, C)
        grad_bary[c, 0] += coeff * np.dot(darea_dpc, B)
        grad_bary[c, 1] += coeff * np.dot(darea_dpc, C)
    
    return grad_bary


def random_bary_init(rng):
    """Generate random barycentric coordinates inside the triangle."""
    bary = np.zeros((N, 2))
    for i in range(N):
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        bary[i] = [u, v]
    return bary


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_bary = None
    best_score = -np.inf
    
    # Multi-start optimization
    n_starts = 50
    tau_schedule = [0.05, 0.03, 0.01, 0.005, 0.002]
    
    for start in range(n_starts):
        if time.time() > deadline:
            break
        
        # Random initialization
        bary0 = random_bary_init(rng)
        
        # Add some symmetry-breaking perturbation
        bary0 += 0.01 * rng.randn(N, 2)
        # Ensure feasibility
        for i in range(N):
            if bary0[i, 0] < 0:
                bary0[i, 0] = 0.01
            if bary0[i, 1] < 0:
                bary0[i, 1] = 0.01
            if bary0[i, 0] + bary0[i, 1] > 1:
                scale = 0.99 / (bary0[i, 0] + bary0[i, 1])
                bary0[i, 0] *= scale
                bary0[i, 1] *= scale
        
        # Optimize with SLSQP using smooth objective
        constraints = [
            {'type': 'ineq', 'fun': lambda b, i=i: b[i, 0] - 0.001},
            {'type': 'ineq', 'fun': lambda b, i=i: b[i, 1] - 0.001},
            {'type': 'ineq', 'fun': lambda b, i=i: 1 - 0.001 - b[i, 0] - b[i, 1]},
        ]
        
        # Build all constraints
        cons = []
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda b, i=i: b[i, 0] - 0.001})
            cons.append({'type': 'ineq', 'fun': lambda b, i=i: b[i, 1] - 0.001})
            cons.append({'type': 'ineq', 'fun': lambda b, i=i: 1 - 0.001 - b[i, 0] - b[i, 1]})
        
        # Use decreasing tau for annealing
        current_bary = bary0.copy()
        for tau in tau_schedule:
            if time.time() > deadline:
                break
            
            def obj(b):
                return soft_min_objective(b, tau=tau)
            
            def grad(b):
                return soft_min_grad(b, tau=tau)
            
            try:
                res = minimize(
                    obj, current_bary, jac=grad,
                    method='SLSQP',
                    bounds=[(0.001, 0.999)] * (2 * N),
                    constraints=cons,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                if res.success and res.fun < 0:  # fun is -soft_min, so negative means positive soft_min
                    current_bary = res.x
            except Exception:
                pass
        
        # Evaluate exact minimum area
        points = bary_to_cart(current_bary)
        score = min_triangle_area(points)
        
        if score > best_score:
            best_score = score
            best_bary = current_bary.copy()
    
    # Final polish: run a few more SLSQP iterations with very small tau
    if best_bary is not None and time.time() < deadline:
        for tau in [0.001, 0.0005, 0.0001]:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    lambda b: soft_min_objective(b, tau=tau),
                    best_bary,
                    jac=lambda b: soft_min_grad(b, tau=tau),
                    method='SLSQP',
                    bounds=[(0.001, 0.999)] * (2 * N),
                    constraints=cons,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                if res.success:
                    points = bary_to_cart(res.x)
                    score = min_triangle_area(points)
                    if score > best_score:
                        best_score = score
                        best_bary = res.x
            except Exception:
                pass
    
    if best_bary is None:
        # Fallback to random
        bary = random_bary_init(rng)
        return bary_to_cart(bary)
    
    return bary_to_cart(best_bary)

# EVOLVE-BLOCK-END
