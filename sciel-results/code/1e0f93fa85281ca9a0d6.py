"""Sphere packing in a unit cube: smooth surrogate gradient ascent + polish."""
import time
import numpy as np

N = 50
_DEADLINE = 150.0 - 15.0          # safety margin before the 150 s kill
_T_START = 10.0                   # initial inverse temperature (beta)
_T_END = 500.0                    # final inverse temperature
_MAX_STEPS = 5000                 # steps per optimization run
_LEARNING_RATE = 0.005


def _score(pts):
    """Largest common radius (higher = better)."""
    n = pts.shape[0]
    # Compute pairwise distances
    d = np.zeros((n, n))
    for i in range(n):
        d[i, i + 1:] = np.linalg.norm(pts[i] - pts[i + 1:], axis=1)
    d = np.maximum(d, d.T)
    r = 0.5 * np.min(d[np.triu_indices(n, k=1)])
    # Wall distance
    wall_d = 0.5 - np.max(np.abs(pts))
    return min(r, wall_d)


def _smooth_objective_and_grad(pts, beta):
    """
    Compute the gradient of the smooth surrogate objective:
    F = -log(sum_{i<j} exp(-beta * d_ij)) - log(sum_i exp(-beta * d_i_wall))
    Maximizing F is equivalent to maximizing the soft-min distance.
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf) # Avoid self-interaction
    
    # Weights for pair interactions: w_ij = exp(-beta * d_ij) / sum(exp(-beta * d_kl))
    # To avoid underflow/overflow, we can shift by min, but with beta up to 500 and d ~ 0.1, exp(-50) is tiny.
    # Actually, we just need the gradient direction.
    # grad F = sum_{j != i} (w_ij / W) * (c_i - c_j) / d_ij * beta ? No.
    # Let S_pair = sum_{i<j} exp(-beta d_ij).
    # d/dc_i log(S_pair) = sum_{j!=i} (exp(-beta d_ij) * (-beta) * (c_i - c_j)/d_ij) / S_pair
    # We want to maximize F, so we ascend this gradient.
    
    exp_pair = np.exp(-beta * d)
    np.fill_diagonal(exp_pair, 0)
    S_pair = np.sum(exp_pair) / 2.0 # Each pair counted twice in sum over all i,j
    
    # Gradient contribution from pairs
    # term_ij = exp(-beta d_ij) * (c_i - c_j) / d_ij
    # grad_i = -beta * sum_{j!=i} term_ij / S_pair
    # Note: (c_i - c_j)/d_ij is the unit vector from j to i.
    # If we move i in direction of (c_i - c_j), d_ij increases.
    # We want to increase d_ij, so we should move in direction of (c_i - c_j).
    # The derivative of log(exp(-beta d)) is -beta * (c_i - c_j)/d.
    # So d/dc_i log S = sum (-beta * (c_i - c_j)/d * exp(-beta d)) / S.
    # We want to maximize log S? No, S is sum of exp(-beta d). As d increases, exp(-beta d) decreases.
    # So maximizing log S means minimizing d? 
    # Wait. The objective is to maximize the minimum distance.
    # Soft min is usually -1/beta log sum exp(-beta d).
    # Let G = -1/beta log S_pair.
    # dG/dc_i = -1/beta * (1/S) * dS/dc_i
    # dS/dc_i = sum_{j!=i} exp(-beta d_ij) * (-beta) * (c_i - c_j)/d_ij
    # dG/dc_i = -1/beta * (1/S) * sum (-beta) * exp(-beta d_ij) * (c_i - c_j)/d_ij
    # dG/dc_i = (1/S) * sum exp(-beta d_ij) * (c_i - c_j)/d_ij
    # This is the direction to increase the soft-min distance.
    
    unit_vec = diff / np.maximum(d, 1e-12)[:,:,None]
    # Zero out diagonal
    np.fill_diagonal(unit_vec, 0)
    
    # Weighted sum of unit vectors
    # grad_pair_i = sum_{j!=i} exp(-beta d_ij) * unit_vec[i,j] / S_pair
    grad_pair = np.sum(exp_pair[:,:,None] * unit_vec, axis=1) / S_pair
    
    # Wall interactions
    # d_i_wall = min(0.5 - |c_ix|, 0.5 - |c_iy|, 0.5 - |c_iz|)
    # Soft wall: -1/beta log sum_k exp(-beta * (0.5 - |c_ik|))
    # Let S_wall_i = sum_k exp(-beta * (0.5 - |c_ik|))
    # dG_wall/dc_ik = (1/S_wall_i) * exp(-beta * (0.5 - |c_ik|)) * (-beta) * d(0.5 - |c_ik|)/dc_ik
    # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
    # So dG_wall/dc_ik = (1/S_wall_i) * exp(-beta * (0.5 - |c_ik|)) * beta * sign(c_ik)
    # Wait, we want to maximize distance to wall.
    # Distance is 0.5 - |c_ik|.
    # If c_ik > 0, moving c_ik down increases distance. Gradient of distance is -1.
    # Let's check signs.
    # G_wall = -1/beta log sum exp(-beta d_wall).
    # d/dc G_wall = -1/beta * (1/S) * sum exp(-beta d_k) * (-beta) * d(d_k)/dc
    # = (1/S) * sum exp(-beta d_k) * d(d_k)/dc
    # d_k = 0.5 - |c_k|. d(d_k)/dc_k = -sign(c_k).
    # So grad = (1/S) * sum exp(-beta d_k) * (-sign(c_k)).
    # If c_k > 0, sign=1, grad = -exp(...)/S. Move in -x direction. Correct.
    
    abs_pts = np.abs(pts)
    d_wall = 0.5 - abs_pts
    exp_wall = np.exp(-beta * d_wall)
    S_wall = np.sum(exp_wall, axis=1, keepdims=True)
    
    # grad_wall_i_k = exp_wall[i,k] * (-sign(pts[i,k])) / S_wall[i]
    signs = np.sign(pts)
    grad_wall = - (exp_wall / S_wall) * signs
    
    total_grad = grad_pair + grad_wall
    return total_grad


def _optimize(pts, t0):
    """Run gradient ascent with annealing."""
    pts = pts.copy()
    current_r = _score(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    # Annealing schedule
    steps = _MAX_STEPS
    # We will check time in the loop
    
    for i in range(steps):
        if time.time() > _DEADLINE:
            break
        
        # Linear annealing of beta
        beta = _T_START + (_T_END - _T_START) * (i / steps)
        
        grad = _smooth_objective_and_grad(pts, beta)
        
        # Adaptive step size? Or fixed?
        # The magnitude of grad is roughly 1/beta? No, grad of log-sum-exp is bounded.
        # Let's use a fixed small step size, maybe scaled by 1/beta?
        # As beta increases, the landscape gets sharper. We should reduce step size.
        step_size = _LEARNING_RATE / (1.0 + beta/100.0)
        
        pts += step_size * grad
        
        # Evaluate score periodically
        if i % 50 == 0:
            r = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    # Final check
    r = _score(pts)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
        
    return best_pts, best_r


def _generate_fcc():
    """Generate FCC lattice points scaled to fit in cube."""
    # FCC basis vectors
    a = 1.0 # lattice constant
    # Points in unit cell: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # We need to fill a cube.
    # Let's generate a 4x4x4 grid of FCC cells? 
    # Or just generate points and select 50 closest to center?
    # Better: Generate a dense grid of FCC points and take the first 50?
    # Or optimize the scale.
    
    # Let's generate points in a 3x3x3 array of FCC cells to get ~ 27*4 = 108 points.
    # Then select 50.
    # FCC unit cell has 4 atoms.
    # To get 50, we need ~ 12.5 cells. A 3x3x3 grid is 27 cells -> 108 atoms.
    # Let's generate a 3x3x3 grid of FCC unit cells.
    
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                # Shift by lattice vector
                offset = np.array([x, y, z])
                # FCC basis
                basis = [
                    [0, 0, 0],
                    [0.5, 0.5, 0],
                    [0.5, 0, 0.5],
                    [0, 0.5, 0.5]
                ]
                for b in basis:
                    p = offset + np.array(b)
                    points.append(p)
                    
    pts = np.array(points)
    # Center the grid
    # The grid ranges from 0 to 3. Center is 1.5.
    pts -= 1.5
    
    # Scale to fit in [-0.5, 0.5]
    # Max coordinate is approx 1.5 + 0.5 = 2.0? 
    # Let's check max abs.
    max_abs = np.max(np.abs(pts))
    # We want max_abs * scale <= 0.45 (leave margin)
    scale = 0.45 / max_abs
    pts *= scale
    
    # Select 50 points. Randomly? Or closest to center?
    # Closest to center is a good heuristic for packing.
    dists = np.sum(pts**2, axis=1)
    indices = np.argsort(dists)[:N]
    return pts[indices]


def _generate_bcc():
    """Generate BCC lattice points scaled to fit in cube."""
    points = []
    # BCC basis: (0,0,0), (0.5, 0.5, 0.5)
    for x in range(4):
        for y in range(4):
            for z in range(4):
                offset = np.array([x, y, z])
                basis = [
                    [0, 0, 0],
                    [0.5, 0.5, 0.5]
                ]
                for b in basis:
                    p = offset + np.array(b)
                    points.append(p)
                    
    pts = np.array(points)
    # Grid 0..4, center 2.0
    pts -= 2.0
    max_abs = np.max(np.abs(pts))
    scale = 0.45 / max_abs
    pts *= scale
    
    dists = np.sum(pts**2, axis=1)
    indices = np.argsort(dists)[:N]
    return pts[indices]


def _polish(pts):
    """High-precision polish using the smooth objective with high beta."""
    best = pts.copy()
    best_r = _score(pts)
    # Use high beta for sharpness
    beta = _T_END
    step = 1e-4
    for it in range(1000):
        if time.time() > _DEADLINE:
            break
        grad = _smooth_objective_and_grad(pts, beta)
        pts += step * grad
        r = _score(pts)
        if r > best_r:
            best_r = r
            best = pts.copy()
        step *= 0.99
    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best = None
    best_r = -1.0

    def try_start(pts):
        nonlocal best, best_r
        pts = pts.copy()
        # Jitter slightly to break symmetry
        pts += rng.normal(0, 0.01, pts.shape)
        pts = np.clip(pts, -0.49, 0.49)
        
        res_pts, res_r = _optimize(pts, t0)
        if res_r > best_r:
            best_r = res_r
            best = res_pts

    # 1) FCC Start
    if time.time() < _DEADLINE:
        try_start(_generate_fcc())

    # 2) BCC Start
    if time.time() < _DEADLINE:
        try_start(_generate_bcc())

    # 3) Random Starts
    for restart in range(5):
        if time.time() > _DEADLINE:
            break
        pts = (rng.random((N, 3)) - 0.5) * 0.95
        try_start(pts)

    # 4) Grid Start
    if time.time() < _DEADLINE:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N].copy()
        try_start(grid)

    # 5) Polish
    if best is not None and time.time() < _DEADLINE:
        polished, pr = _polish(best)
        if pr > best_r:
            best, best_r = polished, pr

    if best is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]

    # guarantee validity
    best = np.clip(best, -0.4999999, 0.4999999)
    return best
