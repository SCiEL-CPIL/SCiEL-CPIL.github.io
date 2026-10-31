"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all C(11,3)=165 triangle index triplets
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def calc_areas_bary(bary):
    """
    Compute all 165 triangle areas from barycentric coordinates.
    bary: (N, 3) array where each row sums to 1 and entries >= 0.
    Area of triangle (i,j,k) in unit equilateral triangle = (sqrt(3)/4) * |det of bary differences|
    But actually for unit area triangle, the area ratio is just the absolute determinant.
    Let me think: if the triangle has area A, and we use barycentric coords,
    the area of sub-triangle is A * |det|. Since our triangle has area 1,
    area = |det|.
    
    Actually the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2) has area sqrt(3)/4.
    The task says "unit-area triangle" but uses this specific triangle.
    Let me re-read: "Place 11 points on or inside a unit-area triangle... The notebook uses an equilateral triangle and divides by its area."
    
    So the actual triangle is (0,0), (1,0), (0.5, sqrt(3)/2) which has area sqrt(3)/4.
    The evaluator measures the raw value. Let me just compute areas in Cartesian and that's what matters.
    
    For barycentric: point = a*V0 + b*V1 + c*V2 where a+b+c=1.
    The area of triangle formed by three points with barycentric coords can be computed as:
    area = (sqrt(3)/4) * |det of the 3x3 matrix with rows being the barycentric coords|
    
    Wait, let me verify. If p1, p2, p3 have barycentric coords (a1,b1,c1), (a2,b2,c2), (a3,b3,c3),
    then the area of triangle p1p2p3 relative to the reference triangle is |det|.
    So absolute area = (sqrt(3)/4) * |det|.
    
    Since we're maximizing the minimum area, and the factor sqrt(3)/4 is constant,
    we can just maximize the minimum |det| and multiply by sqrt(3)/4 at the end.
    But the evaluator measures "raw value" which is the actual area. Let me just compute in Cartesian
    for the final answer, but use barycentric for optimization.
    
    Actually, let me just compute areas directly from barycentric:
    area = (sqrt(3)/4) * |det|
    """
    pi = bary[_TRI_IDX[:, 0]]
    pj = bary[_TRI_IDX[:, 1]]
    pk = bary[_TRI_IDX[:, 2]]
    
    # det of [[a1,b1,c1],[a2,b2,c2],[a3,b3,c3]]
    # = a1*(b2*c3 - c2*b3) - b1*(a2*c3 - c2*a3) + c1*(a2*b3 - b2*a3)
    det = (pi[:,0]*(pj[:,1]*pk[:,2] - pj[:,2]*pk[:,1])
          - pi[:,1]*(pj[:,0]*pk[:,2] - pj[:,2]*pk[:,0])
          + pi[:,2]*(pj[:,0]*pk[:,1] - pj[:,1]*pk[:,0]))
    
    return (np.sqrt(3)/4) * np.abs(det)

def clamp_bary(bary):
    """Clamp barycentric coordinates to valid simplex."""
    bary = np.maximum(bary, 0.0)
    bary = bary / bary.sum(axis=1, keepdims=True)
    return bary

def bary_to_cart(bary):
    """Convert barycentric to Cartesian coordinates."""
    cart = bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2
    return cart

def min_area_from_bary(bary):
    """Minimum area from barycentric coordinates."""
    areas = calc_areas_bary(bary)
    return float(np.min(areas))

def soft_min_obj(flat_bary, beta=100.0):
    """Soft-min objective in barycentric space. Returns negative soft-min area."""
    bary = flat_bary.reshape(-1, 3)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -400, 400)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return -(m - log_sum / beta)

def hard_min_obj(flat_bary):
    """Hard min objective in barycentric space. Returns negative min area."""
    bary = flat_bary.reshape(-1, 3)
    bary = clamp_bary(bary)
    return -min_area_from_bary(bary)

def generate_hexagonal_init():
    """Generate a hexagonal/symmetric initialization in barycentric coordinates."""
    # Start with a structured pattern
    bary = np.zeros((N, 3))
    
    # Center point
    bary[0] = [1/3, 1/3, 1/3]
    
    # 6 points in a ring around center
    cx, cy, cz = 1/3, 1/3, 1/3
    r = 0.2
    for i in range(6):
        angle = i * np.pi / 3
        # In barycentric, we need to be careful. Let's use Cartesian and convert back.
        # Actually, let me place points in Cartesian first then convert.
        pass
    
    # Let me just create a reasonable structured init in Cartesian and convert
    cart = np.zeros((N, 2))
    cart[0] = [0.5, H/3]  # center
    
    # 6 points in hexagonal pattern
    cx, cy = 0.5, H/3
    r = 0.18
    for i in range(6):
        angle = i * np.pi / 3 + np.pi/6
        cart[i+1] = [cx + r * np.cos(angle), cy + r * np.sin(angle)]
    
    # 4 more points
    cart[7] = [0.25, H/6]
    cart[8] = [0.75, H/6]
    cart[9] = [0.5, 2*H/3]
    cart[10] = [0.5, H/6]
    
    # Convert to barycentric
    bary = np.zeros((N, 3))
    for i in range(N):
        x, y = cart[i]
        c = 2.0 * y / np.sqrt(3)
        b = x - 0.5 * c
        a = 1.0 - b - c
        bary[i] = [a, b, c]
    
    return clamp_bary(bary)

def generate_random_init(seed):
    """Generate random initialization in barycentric coordinates."""
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 3))
    for i in range(N):
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        bary[i] = [1 - r1 - r2, r1, r2]
    return bary

def generate_perturbed_init(base_bary, scale, seed):
    """Generate perturbed initialization."""
    rng = np.random.default_rng(seed)
    bary = base_bary + rng.normal(0, scale, size=base_bary.shape)
    return clamp_bary(bary)

def _lbfgs_polish_bary(bary, beta, maxiter, deadline):
    """L-BFGS-B polish in barycentric space."""
    try:
        res = minimize(
            soft_min_obj, bary.flatten(), args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        opt_bary = clamp_bary(res.x.reshape(-1, 3))
        return opt_bary, min_area_from_bary(opt_bary)
    except Exception:
        return bary, min_area_from_bary(bary)

def _nm_polish_bary(bary, deadline):
    """Nelder-Mead polish in barycentric space."""
    try:
        res = minimize(
            hard_min_obj, bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        opt_bary = clamp_bary(res.x.reshape(-1, 3))
        return opt_bary, min_area_from_bary(opt_bary)
    except Exception:
        return bary, min_area_from_bary(bary)

def _coord_descent_bary(bary, deadline, max_rounds=200):
    """Coordinate descent in barycentric space."""
    current = bary.copy()
    score = min_area_from_bary(current)
    steps = [0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step in steps:
        for _ in range(max_rounds // len(steps)):
            if time.time() > deadline:
                return current, score
            improved = False
            for i in range(N):
                for d0 in [-step, 0, step]:
                    for d1 in [-step, 0, step]:
                        if d0 == 0 and d1 == 0:
                            continue
                        trial = current.copy()
                        trial[i, 0] += d0
                        trial[i, 1] += d1
                        trial[i, 2] -= (d0 + d1)  # keep sum = 1
                        trial = clamp_bary(trial)
                        s = min_area_from_bary(trial)
                        if s > score + 1e-15:
                            current = trial
                            score = s
                            improved = True
            if not improved:
                break
    return current, score

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    def tleft():
        return deadline - time.time()
    
    # Generate diverse initializations
    inits = [generate_hexagonal_init()]
    for s in [1, 7, 13, 29, 42, 100, 200, 300, 500, 700, 900, 1200, 1500, 2000]:
        inits.append(generate_random_init(s))
    
    best_bary = None
    best_score = -1.0
    
    # Phase 1: Multi-start L-BFGS with beta schedule
    betas = [20.0, 50.0, 100.0, 200.0, 400.0]
    for init in inits:
        if tleft() < 15:
            break
        active_betas = betas
        if tleft() < 60:
            active_betas = betas[1:]
        if tleft() < 30:
            active_betas = betas[2:]
            
        for beta in active_betas:
            if tleft() < 5:
                break
            bary, sc = _lbfgs_polish_bary(init, beta, 800, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
    
    if best_bary is None:
        best_bary = generate_hexagonal_init()
        best_score = min_area_from_bary(best_bary)
    
    # Phase 2: Nelder-Mead polish on best
    if tleft() > 15:
        bary, sc = _nm_polish_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
    
    # Phase 3: Perturbation multi-start around best
    rng = np.random.default_rng(99999)
    for trial in range(40):
        if tleft() < 10:
            break
        base_scale = 0.005
        scale = base_scale * (1 + 0.5 * np.sin(trial * 0.5))
        if trial % 5 == 0:
            scale *= 2.0
            
        pert = generate_perturbed_init(best_bary, scale, 99999 + trial)
        
        for beta in [100.0, 200.0]:
            if tleft() < 5:
                break
            bary, sc = _lbfgs_polish_bary(pert, beta, 600, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
        
        if tleft() > 8:
            bary, sc = _nm_polish_bary(pert, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
    
    # Phase 4: Final coordinate descent refinement
    if tleft() > 10:
        bary, sc = _coord_descent_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
    
    # Phase 5: One more NM + CD cycle if time remains
    if tleft() > 15:
        bary, sc = _nm_polish_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
        if tleft() > 8:
            bary, sc = _coord_descent_bary(best_bary, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
    
    # Convert best barycentric to Cartesian
    best_pts = bary_to_cart(best_bary)
    return best_pts
# EVOLVE-BLOCK-END
