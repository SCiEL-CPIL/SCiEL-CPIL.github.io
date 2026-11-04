"""Heilbronn triangles: maximize the smallest triangle area among 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11
DEADLINE = 150.0
SAFETY_MARGIN = 15.0
H = np.sqrt(3) / 2.0

# Precompute all triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
N_TRIPLES = len(_TRIPLES)


def _bary_to_cart(bary):
    """Convert barycentric (n,3) to Cartesian (n,2). bary columns: a,b,c with a+b+c=1."""
    # x = b + 0.5*c, y = H*c
    return np.column_stack([bary[:, 1] + 0.5 * bary[:, 2], H * bary[:, 2]])


def _cart_to_bary(cart):
    """Convert Cartesian (n,2) to barycentric (n,3)."""
    x, y = cart[:, 0], cart[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _project_to_simplex(bary):
    """Project barycentric coords to the simplex {a,b,c >= 0, a+b+c=1}."""
    # Clip and normalize
    b = np.clip(bary, 0.0, None)
    s = b.sum(axis=1, keepdims=True)
    s[s < 1e-15] = 1.0
    return b / s


def _all_dets_from_bary(bary):
    """
    Compute 2*area (determinant) for all triples from barycentric coords.
    In barycentric coords for an equilateral triangle with side 1,
    the area of triangle with barycentric coords (a1,b1,c1), (a2,b2,c2), (a3,b3,c3)
    is 0.5 * |det| * (area of reference triangle).
    
    Actually, let's work in Cartesian for area computation to be safe.
    """
    cart = _bary_to_cart(bary)
    return _all_dets_from_cart(cart)


def _all_dets_from_cart(cart):
    """Compute signed 2*area (determinant) for all triples."""
    # For each triple (i,j,k): det = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
    # Vectorized: for each triple, compute cross product
    i_idx = _TRIPLES[:, 0]
    j_idx = _TRIPLES[:, 1]
    k_idx = _TRIPLES[:, 2]
    
    pi = cart[i_idx]  # (T, 2)
    pj = cart[j_idx]
    pk = cart[k_idx]
    
    dets = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return dets


def _min_area_from_bary(bary):
    """Minimum triangle area (not 2*area) from barycentric coords."""
    cart = _bary_to_cart(bary)
    dets = _all_dets_from_cart(cart)
    areas = 0.5 * np.abs(dets)
    return np.min(areas)


def _soft_min(dets, beta):
    """Soft minimum of |dets|/2 (i.e., areas). Returns soft_min value."""
    areas = 0.5 * np.abs(dets)
    # soft_min = -1/beta * log(sum(exp(-beta * a_i)))
    neg = -beta * areas
    max_val = np.max(neg)
    log_sum = max_val + np.log(np.sum(np.exp(neg - max_val)))
    return -log_sum / beta


def _soft_min_grad_weights(dets, beta):
    """Return weights (probs) for soft min gradient w.r.t. areas."""
    areas = 0.5 * np.abs(dets)
    neg = -beta * areas
    max_val = np.max(neg)
    exps = np.exp(neg - max_val)
    return exps / np.sum(exps)


def _objective_and_grad_cart(cart, beta):
    """
    Objective: minimize -soft_min(areas).
    Returns (value, gradient w.r.t. cart (n,2)).
    """
    dets = _all_dets_from_cart(cart)
    areas = 0.5 * np.abs(dets)
    soft_val = _soft_min(dets, beta)
    obj = -soft_val
    
    # Gradient
    weights = _soft_min_grad_weights(dets, beta)  # d(soft_min)/d(area_i)
    # d(obj)/d(area_i) = -weights[i]
    # d(area_i)/d(point) = 0.5 * sign(det_i) * d(det_i)/d(point)
    
    grad = np.zeros_like(cart)
    i_idx = _TRIPLES[:, 0]
    j_idx = _TRIPLES[:, 1]
    k_idx = _TRIPLES[:, 2]
    
    for t in range(N_TRIPLES):
        w = weights[t]
        if w < 1e-15:
            continue
        i, j, k = i_idx[t], j_idx[t], k_idx[t]
        det = dets[t]
        sign = 1.0 if det >= 0 else -1.0
        factor = -w * 0.5 * sign
        
        pi = cart[i]
        pj = cart[j]
        pk = cart[k]
        
        # d(det)/dpi = (pj_y - pk_y, pk_x - pj_x)
        # d(det)/dpj = (pk_y - pi_y, -(pk_x - pi_x))
        # d(det)/dpk = (-(pj_y - pi_y), pj_x - pi_x)
        
        grad[i, 0] += factor * (pj[1] - pk[1])
        grad[i, 1] += factor * (pk[0] - pj[0])
        grad[j, 0] += factor * (pk[1] - pi[1])
        grad[j, 1] += factor * (-(pk[0] - pi[0]))
        grad[k, 0] += factor * (-(pj[1] - pi[1]))
        grad[k, 1] += factor * (pj[0] - pi[0])
    
    return obj, grad


def _objective_bary(bary, beta):
    """Objective in barycentric space (for Nelder-Mead)."""
    bary = _project_to_simplex(bary)
    cart = _bary_to_cart(bary)
    dets = _all_dets_from_cart(cart)
    areas = 0.5 * np.abs(dets)
    return -_soft_min(dets, beta)


def _local_optimize_nm(bary_init, beta, max_iter=2000, time_limit=20.0):
    """Nelder-Mead optimization in barycentric space."""
    bary_init = _project_to_simplex(bary_init.copy())
    
    def obj(x):
        return _objective_bary(x, beta)
    
    result = minimize(
        obj, bary_init, method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    return _project_to_simplex(result.x)


def _local_optimize_lbfgs(bary_init, beta, max_iter=500, time_limit=15.0):
    """L-BFGS-B in Cartesian space with projection."""
    cart_init = _bary_to_cart(_project_to_simplex(bary_init.copy()))
    
    def obj(x):
        pts = x.reshape(-1, 2)
        bary = _cart_to_bary(pts)
        bary = _project_to_simplex(bary)
        pts = _bary_to_cart(bary)
        val, _ = _objective_and_grad_cart(pts, beta)
        return val
    
    def grad(x):
        pts = x.reshape(-1, 2)
        bary = _cart_to_bary(pts)
        bary = _project_to_simplex(bary)
        pts = _bary_to_cart(bary)
        _, g = _objective_and_grad_cart(pts, beta)
        return g.flatten()
    
    result = minimize(
        obj, cart_init.flatten(), jac=grad, method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * (2 * N),
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    pts = result.x.reshape(-1, 2)
    bary = _cart_to_bary(pts)
    bary = _project_to_simplex(bary)
    return bary


def _simulated_annealing(bary_init, beta, n_iter=5000, t_start=0.05, t_end=0.001, time_limit=30.0):
    """Simulated annealing in barycentric space."""
    bary = _project_to_simplex(bary_init.copy())
    best_bary = bary.copy()
    best_val = _min_area_from_bary(bary)
    current_val = _min_area_from_bary(bary)
    
    start_time = time.time()
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        t = t_start * (t_end / t_start) ** (it / max(n_iter - 1, 1))
        
        # Random perturbation
        bary_new = bary + np.random.randn(N, 3) * t * 0.1
        bary_new = _project_to_simplex(bary_new)
        
        new_val = _min_area_from_bary(bary_new)
        
        if new_val >= current_val or np.random.random() < np.exp((new_val - current_val) / (t * 0.01)):
            bary = bary_new
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_bary = bary_new.copy()
    
    return best_bary, best_val


def _structured_init_fps():
    """Farthest Point Sampling based structured initialization."""
    rng = np.random.RandomState(123)
    # Start with centroid
    bary = np.zeros((N, 3))
    bary[0] = [1/3, 1/3, 1/3]
    
    for i in range(1, N):
        # Find point farthest from existing set (using Cartesian distance)
        cart_existing = _bary_to_cart(bary[:i])
        best_dist = -1
        best_pt = None
        for _ in range(500):
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                pt = [u + 0.5 * v, H * v]
                dists = np.min(np.sum((cart_existing - np.array(pt))**2, axis=1))
                if dists > best_dist:
                    best_dist = dists
                    best_pt = pt
        if best_pt is not None:
            bary[i] = _cart_to_bary(np.array([best_pt]))[0]
    
    return bary


def _random_init_bary(rng, n=1):
    """Random barycentric initializations."""
    inits = []
    for _ in range(n):
        b = np.random.dirichlet(np.ones(3), size=N, random_state=rng)
        inits.append(b)
    return inits


def _ga_population_init(rng, pop_size=20):
    """Generate diverse initial population."""
    pop = []
    # Structured
    pop.append(_structured_init_fps())
    # Random
    for _ in range(pop_size - 1):
        pop.append(np.random.dirichlet(np.ones(3), size=N, random_state=rng))
    return pop


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_bary = None
    best_score = -float('inf')
    
    def _update_best(bary, score):
        nonlocal best_bary, best_score
        if score > best_score:
            best_score = score
            best_bary = bary.copy()
    
    def _time_ok():
        return time.time() < deadline
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Generate diverse initial population
    population = _ga_population_init(rng, pop_size=15)
    
    # Evaluate initial population
    for bary in population:
        if not _time_ok():
            break
        score = _min_area_from_bary(bary)
        _update_best(bary, score)
    
    # Phase 2: Local optimization from each initial point with moderate beta
    beta_values = [50.0, 100.0, 200.0]
    
    for beta in beta_values:
        if not _time_ok():
            break
        
        for bary_init in population[:8]:  # Limit to avoid timeout
            if not _time_ok():
                break
            
            # L-BFGS-B
            try:
                bary_opt = _local_optimize_lbfgs(bary_init, beta, max_iter=300)
                score = _min_area_from_bary(bary_opt)
                _update_best(bary_opt, score)
            except Exception:
                pass
            
            if not _time_ok():
                break
            
            # Nelder-Mead
            try:
                bary_nm = _local_optimize_nm(bary_opt if bary_opt is not None else bary_init, beta, max_iter=1500)
                score = _min_area_from_bary(bary_nm)
                _update_best(bary_nm, score)
            except Exception:
                pass
    
    # Phase 3: Simulated annealing from best and a few others
    sa_starts = [best_bary] if best_bary is not None else []
    sa_starts.extend(population[:3])
    
    for bary_start in sa_starts:
        if not _time_ok():
            break
        
        try:
            bary_sa, score_sa = _simulated_annealing(
                bary_start, beta=100.0, n_iter=3000,
                t_start=0.05, t_end=0.001, time_limit=25.0
            )
            _update_best(bary_sa, score_sa)
        except Exception:
            pass
    
    # Phase 4: Final polish with high beta
    if best_bary is not None and _time_ok():
        for beta in [500.0, 1000.0, 2000.0]:
            if not _time_ok():
                break
            try:
                bary_polish = _local_optimize_lbfgs(best_bary, beta, max_iter=500)
                score = _min_area_from_bary(bary_polish)
                _update_best(bary_polish, score)
                
                # NM polish
                bary_polish2 = _local_optimize_nm(bary_polish, beta, max_iter=2000)
                score2 = _min_area_from_bary(bary_polish2)
                _update_best(bary_polish2, score2)
            except Exception:
                pass
    
    # Phase 5: One more round of SA + polish if time remains
    if best_bary is not None and _time_ok():
        try:
            bary_sa2, score_sa2 = _simulated_annealing(
                best_bary, beta=200.0, n_iter=4000,
                t_start=0.03, t_end=0.0005, time_limit=20.0
            )
            _update_best(bary_sa2, score_sa2)
        except Exception:
            pass
        
        if _time_ok():
            try:
                bary_final = _local_optimize_lbfgs(best_bary, beta=2000.0, max_iter=1000)
                score = _min_area_from_bary(bary_final)
                _update_best(bary_final, score)
            except Exception:
                pass
    
    if best_bary is None:
        # Fallback
        best_bary = np.random.dirichlet(np.ones(3), size=N, random_state=np.random.RandomState(7))
    
    # Convert to Cartesian and return
    cart = _bary_to_cart(_project_to_simplex(best_bary))
    return cart


if __name__ == '__main__':
    pts = construct()
    print(f"Points:\n{pts}")
    print(f"Min area: {_min_area_from_bary(_cart_to_bary(pts))}")
