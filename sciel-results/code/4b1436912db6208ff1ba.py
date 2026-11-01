import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

N = 14
D = 3
DEADLINE = 145.0  # Leave 5s safety margin for 150s limit
_t0 = None

def _true_score(pts):
    """Calculate the squared ratio of max to min pairwise distance."""
    if pts.shape != (N, D):
        return np.inf
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _init_tetrahedron():
    """
    Initialize a regular tetrahedron centered at the origin with edge length 1.
    Vertices are at distance 1/sqrt(6) from the center.
    """
    # Standard tetrahedron vertices scaled to unit edge length
    # Edge length of (1,1,1), (-1,-1,1), (-1,1,-1), (1,-1,-1) is 2*sqrt(2).
    # We want edge length 1, so scale by 1/(2*sqrt(2)).
    v = np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1]
    ])
    v /= (2 * np.sqrt(2))
    return v

def _generate_stella_octangula(theta, phi, psi, tx, ty, tz, scale1, scale2):
    """
    Generate points for two interlocking tetrahedra.
    Tetrahedron 1 is fixed at origin with scale scale1.
    Tetrahedron 2 is rotated by (theta, phi, psi) and translated by (tx, ty, tz) with scale scale2.
    """
    t1 = _init_tetrahedron() * scale1
    
    r = Rotation.from_euler('xyz', [theta, phi, psi])
    t2_base = _init_tetrahedron() * scale2
    t2 = r.apply(t2_base)
    t2 = t2 + np.array([tx, ty, tz])
    
    pts = np.vstack([t1, t2])
    return pts

def _obj(x):
    """Objective function for differential evolution."""
    theta, phi, psi, tx, ty, tz, scale1, scale2 = x
    
    # Constrain scales to be positive and reasonable
    # If scales are too small, ratio is undefined or points coincide.
    # We want to explore the relative sizes.
    s1 = abs(scale1)
    s2 = abs(scale2)
    
    if s1 < 0.1 or s2 < 0.1:
        return 1e6
        
    pts = _generate_stella_octangula(theta, phi, psi, tx, ty, tz, s1, s2)
    
    # Center the configuration to remove translational invariance
    # This helps with numerical stability and ensures the "center" is well-defined
    pts = pts - pts.mean(axis=0)
    
    return _true_score(pts)

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
        
    best_pts = None
    best_score = np.inf
    
    def consider(pts):
        nonlocal best_pts, best_score
        if pts is None:
            return
        if pts.shape != (N, D):
            return
        pts = pts - pts.mean(axis=0)
        s = _true_score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts.copy()
            return True
        return False

    # 1. Initial guesses
    # Guess 1: Perfect Stella Octangula (scales equal, no rotation, no translation)
    # This is the standard configuration where they interlock perfectly.
    # Rotation of 180 deg around any axis maps one tet to the other?
    # Actually, the dual tetrahedron is rotated by 180 degrees around any of the 4 axes passing through midpoints of opposite edges? No.
    # The vertices of the second tetrahedron are the midpoints of the faces of the first? No, that's the medial tetrahedron.
    # The stella octangula consists of two tetrahedra such that their union is a cuboctahedron (if scaled correctly) or just interlocked.
    # A regular tetrahedron's dual is another regular tetrahedron.
    # Let's try a few standard symmetric configurations.
    
    # Config 1: Identity (they overlap completely, bad)
    # Config 2: Rotated 180 deg around Z axis?
    # Config 3: Rotated 180 deg around axis (1,1,1)?
    
    # Let's just use DE to find the best rotation/translation/scale.
    # But we need a good start.
    # The "dual" tetrahedron of a regular tetrahedron centered at origin is obtained by scaling by -1/3? No.
    # The vertices of the dual are normal to the faces.
    # For our unit edge tetrahedron, the dual is also a regular tetrahedron.
    # Let's just let DE explore.
    
    # Initial point for DE:
    # theta, phi, psi in [0, 2pi]
    # tx, ty, tz in [-1, 1]
    # scale1, scale2 in [0.5, 2.0]
    
    bounds = [
        (0, 2 * np.pi),
        (0, 2 * np.pi),
        (0, 2 * np.pi),
        (-1.5, 1.5),
        (-1.5, 1.5),
        (-1.5, 1.5),
        (0.5, 2.0),
        (0.5, 2.0)
    ]
    
    start_time = time.time()
    remaining = DEADLINE - (start_time - _t0)
    
    # Run Differential Evolution
    if remaining > 10:
        def callback(xk, convergence):
            return time.time() - _t0 > DEADLINE - 15
        
        try:
            # Use a larger population to explore the rotation space
            result = differential_evolution(
                _obj, 
                bounds, 
                maxiter=100, 
                popsize=30, 
                tol=1e-5, 
                seed=42,
                callback=callback,
                polish=True
            )
            if result.fun < best_score:
                # Reconstruct points from best parameters
                x_best = result.x
                theta, phi, psi, tx, ty, tz, s1, s2 = x_best
                pts_de = _generate_stella_octangula(theta, phi, psi, tx, ty, tz, abs(s1), abs(s2))
                consider(pts_de)
        except Exception:
            pass

    # If DE failed or found a bad score, try a few specific symmetric configurations
    if best_score > 2.0 or best_pts is None:
        # Try the "perfect" stella octangula
        # The second tetrahedron is the dual of the first.
        # For a regular tetrahedron, the dual is similar and rotated.
        # One known configuration: scale 1, rotation 180 deg around (1,1,1)?
        # Let's try rotating 180 degrees around the x-axis.
        r_x = Rotation.from_euler('x', np.pi)
        t1 = _init_tetrahedron()
        t2 = r_x.apply(_init_tetrahedron())
        pts_sym1 = np.vstack([t1, t2])
        consider(pts_sym1)
        
        # Try rotating 180 degrees around y-axis
        r_y = Rotation.from_euler('y', np.pi)
        t2 = r_y.apply(_init_tetrahedron())
        pts_sym2 = np.vstack([t1, t2])
        consider(pts_sym2)
        
        # Try rotating 180 degrees around z-axis
        r_z = Rotation.from_euler('z', np.pi)
        t2 = r_z.apply(_init_tetrahedron())
        pts_sym3 = np.vstack([t1, t2])
        consider(pts_sym3)

        # Try the dual tetrahedron (inverted and scaled)
        # The dual of a regular tetrahedron is congruent to it.
        # The vertices of the dual are opposite to the faces.
        # For our tetrahedron, the dual is just -t1 scaled?
        # Actually, the stella octangula is formed by two tetrahedra of the same size.
        # One is rotated by 180 degrees around an axis passing through the midpoints of two opposite edges?
        # No, the axis is through the center and the midpoint of an edge?
        # Let's just trust DE found something if it did.
        # If not, the symmetric guesses above are decent.

    # Local polish with Nelder-Mead on the 9 parameters
    if best_pts is not None:
        current_time = time.time()
        remaining = DEADLINE - (current_time - _t0)
        
        if remaining > 5:
            from scipy.optimize import minimize
            
            # We need to map the 9 parameters to the objective.
            # We'll optimize the parameters directly.
            # But we don't know the parameters that generated best_pts.
            # We can try to recover them or just polish the points directly?
            # Polishing points directly (42 dims) is what the previous program did.
            # Since we are in a new structural family, maybe we should polish the 9 params?
            # But we don't have the 9 params for best_pts if it came from a symmetric guess.
            # If it came from DE, we have result.x.
            
            # Let's assume best_pts came from DE. If not, we can't easily polish the 9 params.
            # However, we can just do a standard Nelder-Mead on the 42 coordinates starting from best_pts.
            # This is a safe fallback that might improve the specific configuration found.
            
            def _nm_obj(x):
                pts = x.reshape(N, D)
                pts = pts - pts.mean(axis=0)
                s = _true_score(pts)
                # Regularize to keep scale reasonable
                norm = np.linalg.norm(pts)
                if norm < 1e-6:
                    return 1e6
                return s

            try:
                res_nm = minimize(
                    _nm_obj, 
                    best_pts.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res_nm.fun < best_score:
                    pts_nm = res_nm.x.reshape(N, D)
                    consider(pts_nm)
            except Exception:
                pass

    if best_pts is None:
        # Fallback: just return the first symmetric guess
        t1 = _init_tetrahedron()
        r_z = Rotation.from_euler('z', np.pi)
        t2 = r_z.apply(_init_tetrahedron())
        best_pts = np.vstack([t1, t2])
        
    return best_pts
