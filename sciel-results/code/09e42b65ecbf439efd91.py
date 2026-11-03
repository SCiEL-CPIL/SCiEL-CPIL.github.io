"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import sys

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def _radius(c):
    """Compute the maximum common radius for centers c of shape (N,2)."""
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(axis=1))
    rp = d.min() / 2.0
    rw = np.minimum(np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
                    np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)).min()
    return min(rp, rw)

def _soft_radius(c, beta=100.0):
    """
    Smooth approximation of the minimum distance.
    Uses log-sum-exp trick for numerical stability.
    Returns a scalar value approximating min(distances).
    """
    n = c.shape[0]
    # Pairwise distances
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d_sq = (diff ** 2).sum(axis=1)
    
    # Wall distances
    # distance to left wall: x + 0.5
    # distance to right wall: 0.5 - x
    # distance to bottom wall: y + 0.5
    # distance to top wall: 0.5 - y
    d_walls = np.concatenate([
        c[:, 0] + 0.5,
        0.5 - c[:, 0],
        c[:, 1] + 0.5,
        0.5 - c[:, 1]
    ])
    
    # We want to maximize the minimum of (d_pairs/2, d_walls).
    # Let's work with squared distances to avoid sqrt in gradient, 
    # but soft-min is easier on linear distances. 
    # However, for optimization, minimizing the negative soft-max of distances is standard.
    
    # Combine all distances: pair distances are halved because radius = d/2.
    # Wait, radius r satisfies 2r <= d_pair and r <= d_wall.
    # So we want to maximize r such that r <= d_pair/2 and r <= d_wall.
    # This is equivalent to maximizing min( min(d_pair)/2, min(d_wall) ).
    
    # Let's define the vector of "constrained radii":
    # r_i = d_pair_i / 2 for all pairs
    # r_j = d_wall_j for all walls
    # We want max min(r).
    
    # Vector of values to take soft-min of:
    # v = [d_pair_1/2, ..., d_pair_M/2, d_wall_1, ..., d_wall_K]
    
    v_pairs = np.sqrt(d_sq) / 2.0
    v = np.concatenate([v_pairs, d_walls])
    
    # Soft min: -1/beta * log(sum(exp(-beta * v)))
    # To avoid overflow, shift by max(v) or min(v)?
    # For soft-min, we want the value close to min(v).
    # log-sum-exp for min: min(v) + log(sum(exp(v - min(v)))) / ... no.
    # Standard soft-min: m = -1/beta * log( sum(exp(-beta * v)) )
    # Let v_min = min(v).
    # exp(-beta * v) = exp(-beta * (v - v_min)) * exp(-beta * v_min)
    # sum(exp(-beta * v)) = exp(-beta * v_min) * sum(exp(-beta * (v - v_min)))
    # log(sum) = -beta * v_min + log(sum(exp(-beta * (v - v_min))))
    # m = -1/beta * (-beta * v_min + log(...)) = v_min - 1/beta * log(sum(exp(-beta * (v - v_min))))
    
    v_min = np.min(v)
    # If v_min is very small, exp(-beta*(v-v_min)) is bounded by 1.
    # This is stable.
    
    exp_term = np.exp(-beta * (v - v_min))
    log_sum = np.log(np.sum(exp_term))
    
    return v_min - (1.0 / beta) * log_sum

def _negative_soft_radius(c, beta=100.0):
    return -_soft_radius(c, beta)

def _init_grid():
    """Simple grid initialization."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()

def _init_hexagonal():
    """Hexagonal lattice of ~50 points inside the square."""
    pts = []
    s = 0.14
    xs = np.arange(-0.5 + s / 2, 0.5, s)
    for row, y in enumerate(np.arange(-0.5 + s / 2, 0.5, s * np.sqrt(3) / 2)):
        offset = (s / 2) if row % 2 else 0.0
        for x in xs:
            xx = x + offset
            if -0.5 <= xx <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([xx, y])
    pts = np.array(pts)
    if len(pts) >= N:
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists)[:N]
        return pts[idx].copy()
    else:
        return _init_grid()

def _init_random():
    """Random initialization with some jitter to escape local minima."""
    np.random.seed(42)
    # Start with grid and add noise
    c = _init_grid()
    c += np.random.normal(0, 0.01, c.shape)
    c = np.clip(c, -0.4, 0.4)
    return c

def _optimize_single(c0, t0, deadline, method='L-BFGS-B', beta=100.0):
    """
    Optimize from a single starting point.
    """
    n = c0.shape[0]
    
    # Bounds: centers must be in [-0.5, 0.5]
    # To ensure validity, we can keep them slightly inside, e.g. [-0.49, 0.49]
    # The objective function will push them to maximize radius, so they will naturally
    # move away from walls if possible, but we must prevent them from going outside.
    bounds = [(-0.499, 0.499)] * (2 * n)
    
    # Objective function
    def obj(x):
        c = x.reshape(n, 2)
        # Add a small penalty if any point is outside [-0.5, 0.5] to help SLSQP if we use it
        # But with L-BFGS-B bounds, this is handled.
        return _negative_soft_radius(c, beta)
    
    # Gradient for L-BFGS-B
    def grad(x):
        c = x.reshape(n, 2)
        # Finite difference gradient is slow but robust for prototyping.
        # Given time constraints, let's try analytic or rely on L-BFGS-B's internal FD?
        # L-BFGS-B supports analytic gradients. Let's implement it.
        
        n = c.shape[0]
        # v = [d_pair/2, d_wall]
        # soft_min = v_min - 1/beta * log(sum(exp(-beta * (v - v_min))))
        # d(soft_min)/dx = ... complicated.
        # Let's use a simpler approach: 
        # The gradient of soft-min is a weighted average of the gradients of the individual distances.
        # weights w_k = exp(-beta * (v_k - v_min)) / sum(exp(-beta * (v - v_min)))
        # grad(soft_min) = sum(w_k * grad(v_k))
        
        iu = np.triu_indices(n, k=1)
        diff = c[iu[0]] - c[iu[1]]
        d_sq = (diff ** 2).sum(axis=1)
        d_pairs = np.sqrt(d_sq)
        # Avoid division by zero
        d_pairs_safe = np.where(d_pairs < 1e-10, 1e-10, d_pairs)
        
        v_pairs = d_pairs_safe / 2.0
        
        d_walls = np.concatenate([
            c[:, 0] + 0.5,
            0.5 - c[:, 0],
            c[:, 1] + 0.5,
            0.5 - c[:, 1]
        ])
        
        v = np.concatenate([v_pairs, d_walls])
        v_min = np.min(v)
        exp_term = np.exp(-beta * (v - v_min))
        sum_exp = np.sum(exp_term)
        w = exp_term / sum_exp
        
        # Gradient of v_pairs with respect to c
        # v_pair_k = 0.5 * ||c_i - c_j||
        # d v_pair_k / d c_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # d v_pair_k / d c_j = -0.5 * (c_i - c_j) / ||c_i - c_j||
        
        g = np.zeros((n, 2))
        
        # Pair contributions
        # w[:M] corresponds to pairs
        w_pairs = w[:len(iu[0])]
        
        # Vectorize:
        # For each pair k (i, j), add w_k * 0.5 * (c_i - c_j) / d_k to g[i]
        # and subtract from g[j]
        
        # We can use np.add.at for accumulation
        # diff = c[i] - c[j]
        # unit_diff = diff / d_pairs_safe
        # contribution_i = w_pairs * 0.5 * unit_diff
        # contribution_j = -w_pairs * 0.5 * unit_diff
        
        unit_diff = diff / d_pairs_safe[:, np.newaxis]
        contrib = w_pairs[:, np.newaxis] * 0.5 * unit_diff
        
        np.add.at(g, iu[0], contrib)
        np.add.at(g, iu[1], -contrib)
        
        # Wall contributions
        # w[len(M):] corresponds to walls
        w_walls = w[len(iu[0]):]
        
        # Order in d_walls:
        # 0..N-1: c[:,0] + 0.5 (grad wrt c[:,0] is 1)
        # N..2N-1: 0.5 - c[:,0] (grad wrt c[:,0] is -1)
        # 2N..3N-1: c[:,1] + 0.5 (grad wrt c[:,1] is 1)
        # 3N..4N-1: 0.5 - c[:,1] (grad wrt c[:,1] is -1)
        
        # Accumulate into g
        # g[:, 0] += w_walls[0:N] * 1
        # g[:, 0] -= w_walls[N:2*N] * 1
        # g[:, 1] += w_walls[2*N:3*N] * 1
        # g[:, 1] -= w_walls[3*N:4*N] * 1
        
        g[:, 0] += w_walls[0:N]
        g[:, 0] -= w_walls[N:2*N]
        g[:, 1] += w_walls[2*N:3*N]
        g[:, 1] -= w_walls[3*N:4*N]
        
        # The objective is -soft_min, so gradient is -grad(soft_min)
        return -g.reshape(-1)
    
    x0 = c0.reshape(-1)
    
    # Check time
    if (time.time() - t0) > deadline - 20:
        return c0
    
    try:
        if method == 'L-BFGS-B':
            res = minimize(obj, x0, jac=grad, bounds=bounds, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
        elif method == 'SLSQP':
            # SLSQP can handle constraints but is slower.
            # Let's stick to L-BFGS-B with bounds for speed.
            res = minimize(obj, x0, bounds=bounds, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12})
        else:
            return c0
            
        c_opt = res.x.reshape(n, 2)
        
        # Clip to ensure strict validity
        c_opt = np.clip(c_opt, -0.5 + 1e-15, 0.5 - 1e-15)
        
        return c_opt
        
    except Exception:
        return c0

def _delaunay_refine(c, t0, deadline):
    """
    Use Delaunay triangulation to identify "tight" edges and perform local refinement.
    This is a structural change: instead of global soft-min, we focus on the contact graph.
    """
    n = c.shape[0]
    c = c.copy()
    
    for _ in range(10):
        if (time.time() - t0) > deadline - 10:
            break
            
        try:
            tri = Delaunay(c)
        except Exception:
            break
            
        # Get edges
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[0], simplex[2]))))
            
        # Find the current radius
        r = _radius(c)
        
        # Identify active edges: those with distance close to 2*r
        # And wall contacts
        active_pairs = []
        for i, j in edges:
            d = np.linalg.norm(c[i] - c[j])
            if d < 2 * r * 1.01:
                active_pairs.append((i, j))
                
        # If no active pairs, we might be in a loose configuration.
        # Just return current c.
        if not active_pairs:
            break
            
        # Try to optimize only the active constraints?
        # This is complex. Let's just use the result from L-BFGS-B which already
        # considers all constraints via soft-min.
        # This function is a placeholder for potential future refinement.
        break
        
    return c

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin for 150s limit
    
    # Multiple initializations to escape local minima
    inits = [
        _init_grid(),
        _init_hexagonal(),
        _init_random()
    ]
    
    # Add a few more random inits if time permits
    np.random.seed(42)
    for i in range(5):
        c_rand = np.random.uniform(-0.4, 0.4, (N, 2))
        inits.append(c_rand)
        
    best_c = inits[0]
    best_r = _radius(best_c)
    
    # Sort inits by initial radius? No, diversity is key.
    # But let's prioritize those with better initial radius to save time.
    # Actually, L-BFGS-B is fast. Let's run all.
    
    for c0 in inits:
        if (time.time() - t0) > deadline - 30:
            break
            
        # Increase beta gradually to sharpen the minimum
        # Start with low beta for global search, end with high beta for precision
        betas = [20.0, 50.0, 100.0, 200.0]
        
        current_c = c0
        for beta in betas:
            if (time.time() - t0) > deadline - 10:
                break
                
            current_c = _optimize_single(current_c, t0, deadline, method='L-BFGS-B', beta=beta)
            
        r = _radius(current_c)
        if r > best_r:
            best_r = r
            best_c = current_c
            
    # Final polish with high beta
    if (time.time() - t0) < deadline - 10:
        best_c = _optimize_single(best_c, t0, deadline, method='L-BFGS-B', beta=500.0)
        
    # Ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-15, 0.5 - 1e-15)
    
    if not np.all(np.isfinite(best_c)) or _radius(best_c) <= 0:
        return _init_grid()
        
    return best_c
# EVOLVE-BLOCK-END
