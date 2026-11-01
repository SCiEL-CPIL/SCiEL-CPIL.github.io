"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place 50 equal circles in a unit circle to maximize the common radius.
    Multi-start Nelder-Mead with smooth potential energy objective.
    """
    
    def objective(x):
        """
        Smooth objective: penalize boundary violations and close pairs.
        """
        pts = x.reshape(N, 2)
        dists = np.linalg.norm(pts, axis=1)
        
        # Boundary penalty: quadratic for exceeding 0.98
        b_pen = np.sum(np.maximum(0, dists - 0.98) ** 2) * 50
        
        # Pairwise: smooth repulsion for close pairs
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Only consider pairs closer than 0.4
        mask = d2 < 0.16
        d = np.sqrt(d2[mask] + 1e-12)
        p_pen = np.sum((0.4 - d) ** 2 / (d + 1e-6))
        
        return b_pen + p_pen
    
    def get_radius(x):
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        return min(np.sqrt(np.min(d2)) / 2, 1 - np.max(d_origin))
    
    def init_hex(scale):
        pts = []
        sp = 0.2 * scale
        for r in range(-7, 8):
            for c in range(-7, 8):
                x = c * sp
                y = r * sp * 0.866
                if r % 2:
                    x += sp * 0.5
                if x*x + y*y < (scale - 0.15)**2:
                    pts.append([x, y])
        rng = np.random.default_rng(99)
        rng.shuffle(pts)
        return np.array(pts[:N]).flatten()
    
    def init_rings():
        pts = [[0, 0]]
        for n, rad in [(6, 0.22), (12, 0.44), (18, 0.66), (14, 0.85)]:
            for i in range(n):
                a = 2*np.pi*i/n
                pts.append([rad*np.cos(a), rad*np.sin(a)])
        rng = np.random.default_rng(7)
        return (np.array(pts[:N]) + rng.uniform(-0.02, 0.02, (N,2))).flatten()
    
    def init_rand(seed):
        rng = np.random.default_rng(seed)
        a = rng.uniform(0, 2*np.pi, N)
        r = (rng.uniform(0, 1, N)**0.5) * 0.88
        return np.column_stack([r*np.cos(a), r*np.sin(a)]).flatten()
    
    inits = [
        init_hex(0.82),
        init_hex(0.88),
        init_rings(),
        init_rand(1),
        init_rand(2),
        init_rand(3),
    ]
    
    best_x, best_r = None, -1
    
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 2500, 'maxfev': 4000,
                                   'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            r = get_radius(res.x)
            if r > best_r:
                best_r, best_x = r, res.x.copy()
        except:
            pass
    
    if best_x is None:
        return init_rand(1).reshape(N, 2)
    
    # Refine with tighter tolerances
    try:
        res2 = minimize(objective, best_x, method='Nelder-Mead',
                       options={'maxiter': 1500, 'maxfev': 2500,
                                'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        r2 = get_radius(res2.x)
        if r2 > best_r:
            best_x = res2.x.copy()
    except:
        pass
    
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
