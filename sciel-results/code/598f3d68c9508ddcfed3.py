import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to minimum pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _gram_objective(g_flat, sharpness=20.0):
    """
    Objective function for optimizing the Gram matrix G.
    G must be PSD and rank <= 3.
    Distance squared D_ij = G_ii + G_jj - 2G_ij.
    We want to minimize (max D_ij / min D_ij)^2.
    
    To handle the non-differentiability of max/min and the constraints:
    1. Soft max/min on the off-diagonal entries of D.
    2. Penalty for negative eigenvalues of G (PSD constraint).
    3. Penalty for eigenvalues beyond the top 3 (Rank constraint).
    """
    G = g_flat.reshape(N, N)
    
    # Enforce symmetry (numerical stability)
    G = (G + G.T) / 2.0
    
    # Compute squared distances
    # D = diag(G) + diag(G).T - 2G
    diag_G = np.diag(G).copy()
    D = diag_G[:, None] + diag_G[None, :] - 2.0 * G
    
    # We only care about off-diagonal elements
    mask = ~np.eye(N, dtype=bool)
    D_off = D[mask]
    
    if len(D_off) == 0:
        return 1e6
        
    d_min = D_off.min()
    d_max = D_off.max()
    
    # If distances are negative or too small, penalize
    if d_min < 1e-6:
        return 1e6 + d_min * 1e3
        
    k = sharpness
    
    # Soft min/max on D_off
    # Shift to avoid overflow
    shift_min = d_min
    shift_max = d_max
    
    exp_neg_k_d = np.exp(-k * (D_off - shift_min))
    soft_min = shift_min - (1.0 / k) * np.log(np.sum(exp_neg_k_d))
    
    exp_k_d = np.exp(k * (D_off - shift_max))
    soft_max = shift_max + (1.0 / k) * np.log(np.sum(exp_k_d))
    
    if soft_min < 1e-9:
        return 1e6
        
    ratio = soft_max / soft_min
    obj_val = ratio ** 2
    
    # Constraints:
    # 1. PSD: All eigenvalues >= 0
    # 2. Rank 3: Only top 3 eigenvalues should be non-zero
    
    try:
        eigvals = np.linalg.eigvalsh(G)
    except np.linalg.LinAlgError:
        return 1e6
        
    # PSD Penalty: sum of negative eigenvalues squared
    neg_eig = eigvals[eigvals < 0]
    psd_penalty = np.sum(neg_eig ** 2) * 100.0
    
    # Rank Penalty: sum of eigenvalues from index 3 onwards squared
    # We want eigvals[3:] to be 0
    rank_penalty = np.sum(eigvals[3:] ** 2) * 100.0
    
    # Trace normalization to prevent scaling issues (optional, but helps)
    # The ratio is scale-invariant, but the optimizer might drift.
    # Let's add a small penalty for trace deviation from a target, e.g., N
    trace_pen = (np.trace(G) - N) ** 2 * 0.1
    
    return obj_val + psd_penalty + rank_penalty + trace_pen

def _optimize_gram(start_G, time_limit):
    """
    Run L-BFGS-B on the Gram matrix.
    """
    start_time = time.time()
    x0 = start_G.flatten()
    
    def obj_flat(x):
        if time.time() - start_time > time_limit:
            return None # Signal timeout
        return _gram_objective(x)
        
    # We need a callback to stop if timeout
    stop_flag = [False]
    def callback(xk):
        if time.time() - start_time > time_limit:
            stop_flag[0] = True
            return True
        return False
        
    try:
        res = minimize(
            _gram_objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5},
            callback=callback
        )
        
        if stop_flag[0]:
            return start_G
            
        return res.x.reshape(N, N)
        
    except Exception:
        return start_G

def _gram_to_points(G):
    """
    Convert a PSD Gram matrix to coordinates via eigen-decomposition.
    """
    # Symmetrize
    G = (G + G.T) / 2.0
    
    try:
        eigvals, eigvecs = np.linalg.eigh(G)
    except np.linalg.LinAlgError:
        return None
        
    # Take top 3 eigenvalues
    idx = np.argsort(eigvals)[::-1][:D]
    top_eigvals = eigvals[idx]
    top_eigvecs = eigvecs[:, idx]
    
    # Clip small negative values to 0
    top_eigvals = np.maximum(top_eigvals, 0)
    
    # Points = V * sqrt(lambda)
    pts = top_eigvecs * np.sqrt(top_eigvals)
    
    # Center the points (remove mean) to ensure rank <= D strictly if needed, 
    # though Gram matrix construction from centered points is standard.
    # Actually, if G is the Gram matrix of centered points, trace is sum of squared norms.
    # Let's just return the points.
    return pts

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = np.inf
    
    # Generate diverse initial Gram matrices
    # We need PSD matrices of rank 3.
    # Strategy: Generate random points X (N x 3), compute G = X X^T.
    # Then perturb X slightly or G slightly and optimize.
    
    inits = []
    
    # 1. Fibonacci Sphere based init
    i = np.arange(N)
    phi = np.arccos(1 - 2*(i+0.5)/N)
    theta = np.pi*(1+5**0.5)*i
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    fib_pts = np.column_stack([x, y, z])
    # Center it
    fib_pts -= fib_pts.mean(axis=0)
    G_fib = fib_pts @ fib_pts.T
    inits.append(G_fib)
    
    # 2. Icosahedron + 2 random
    phi_ratio = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi_ratio, 1], [0, phi_ratio, -1], [0, -phi_ratio, 1], [0, -phi_ratio, -1],
        [1, 0, phi_ratio], [1, 0, -phi_ratio], [-1, 0, phi_ratio], [-1, 0, -phi_ratio],
        [phi_ratio, 1, 0], [phi_ratio, -1, 0], [-phi_ratio, 1, 0], [-phi_ratio, -1, 0]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    icosa_pts = verts[:12].copy()
    extra = rng.normal(0, 1, size=(2, 3))
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    icosa_pts = np.vstack([icosa_pts, extra])
    icosa_pts -= icosa_pts.mean(axis=0)
    G_icosa = icosa_pts @ icosa_pts.T
    inits.append(G_icosa)
    
    # 3. Random Gaussian inits
    for _ in range(10):
        rand_pts = rng.normal(0, 1, size=(N, 3))
        rand_pts -= rand_pts.mean(axis=0)
        G_rand = rand_pts @ rand_pts.T
        inits.append(G_rand)
        
    # Optimization Loop
    for init_G in inits:
        if time.time() - _t0 > DEADLINE - 10:
            break
            
        # Run optimization on Gram matrix
        opt_G = _optimize_gram(init_G, time_limit=3.0)
        
        # Convert back to points
        opt_pts = _gram_to_points(opt_G)
        
        if opt_pts is None or opt_pts.shape != (N, D):
            continue
            
        # Ensure distinct points
        d = cdist(opt_pts, opt_pts)
        np.fill_diagonal(d, np.inf)
        if d.min() < 1e-6:
             # If points collapsed, skip this result
             continue
             
        score = _true_score(opt_pts)
        
        if score < best_score:
            best_score = score
            best_pts = opt_pts.copy()
            
    # If no valid points found, fallback to Fibonacci
    if best_pts is None:
        # Reconstruct from first init
        best_pts = _gram_to_points(inits[0])
        if best_pts is None:
            # Ultimate fallback
            best_pts = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
                                 [0.5,0.5,0.5], [-0.5,-0.5,-0.5], [0.5,-0.5,0.5], [-0.5,0.5,-0.5],
                                 [0.5,0.5,-0.5], [-0.5,-0.5,0.5], [0.1,0.1,0.1], [-0.1,-0.1,-0.1]])
            
    # Final sanity check
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6:
        best_pts += rng.normal(0, 1e-4, size=(N, D))
        
    return best_pts
