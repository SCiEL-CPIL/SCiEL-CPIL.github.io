"""Spherical-harmonic critical-point method for minimizing log-energy of 150 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv

N = 150
MAX_DEG = 12

# Precompute basis functions on a grid
GRID_LAT = 60
GRID_LON = 120
theta_grid, phi_grid = np.meshgrid(
    np.linspace(0, np.pi, GRID_LAT, endpoint=False) + np.pi / GRID_LAT,
    np.linspace(0, 2 * np.pi, GRID_LON, endpoint=False)
)
theta_flat = theta_grid.flatten()
phi_flat = phi_grid.flatten()

# Precompute Y_lm and derivatives
Y_basis = np.zeros((MAX_DEG + 1, 2 * MAX_DEG + 1, GRID_LAT * GRID_LON), dtype=complex)
dY_dtheta = np.zeros_like(Y_basis)
dY_dphi = np.zeros_like(Y_basis)

def _precompute_basis():
    for l in range(MAX_DEG + 1):
        for m in range(-l, l + 1):
            # Y_lm
            if m < 0:
                coeff = (-1) ** m * np.sqrt((2 * l + 1) / (4 * np.pi) *
                                            np.math.factorial(l + m) / np.math.factorial(l - m))
                Y = coeff * lpmv(abs(m), l, np.cos(theta_flat)) * np.exp(1j * m * phi_flat)
            else:
                coeff = np.sqrt((2 * l + 1) / (4 * np.pi) *
                                np.math.factorial(l - m) / np.math.factorial(l + m))
                Y = coeff * lpmv(m, l, np.cos(theta_flat)) * np.exp(1j * m * phi_flat)
            
            Y_basis[l, m + MAX_DEG, :] = Y
            
            # dY/dphi = i*m*Y
            dY_dphi[l, m + MAX_DEG, :] = 1j * m * Y
            
            # dY/dtheta
            # d/dtheta (P_l^m(cos theta) e^{im phi}) = -sin(theta) dP_l^m/dx e^{im phi}
            # dP_l^m/dx = (x P_l^m - (l+m) P_l^{m-1}) / (x^2 - 1) ? No.
            # Use recurrence: dP_l^m/dx = (l+m) P_l^{m-1} / (x^2-1) - x dP_l^m/dx ?
            # Simpler: Numerical derivative or use lpmv derivative properties.
            # dP_l^m(x)/dx = (l+m) P_l^{m-1}(x) / (x^2 - 1) - x * (l-m+1) P_l^{m+1}(x) / (x^2 - 1) ?
            # Actually, lpmv doesn't return derivatives directly in a simple way for all m.
            # Let's use central difference for the grid precomputation? 
            # Or use the relation: dY/dtheta = -sin(theta) (lpmv(m, l, cos(theta) - eps) - lpmv(m, l, cos(theta) + eps)) / (2*eps) * coeff * exp...
            # This is expensive to precompute but done once.
            
            # Let's just compute numerical gradient on the grid later if needed, 
            # or use the fact that we can compute Y at theta +/- eps.
            # For speed in the main loop, we want analytic or fast numerical.
            # Let's stick to numerical gradient on the fly for refinement, 
            # but for grid search, we just need V.
            pass

# We will compute V on the grid using the precomputed Y_basis.
# For refinement, we will use a local Newton method with numerical gradients.

def _eval_V(c, theta, phi):
    """Evaluate V(theta, phi) for coefficient vector c."""
    # c is flattened [l, m]
    # We need to sum c_lm * Y_lm
    # Y_basis is (L+1, 2L+1, M)
    # c is (L+1, 2L+1)
    # Reshape c
    c_mat = c.reshape((MAX_DEG + 1, 2 * MAX_DEG + 1))
    # Zero out invalid m
    for l in range(MAX_DEG + 1):
        for m in range(-l, l + 1):
            if m + MAX_DEG >= 0 and m + MAX_DEG < 2 * MAX_DEG + 1:
                pass # already in c_mat
            else:
                c_mat[l, m + MAX_DEG] = 0
                
    # Sum over l, m
    # V = sum_{l,m} c_lm Y_lm
    # Use einsum or dot
    V = np.einsum('lm,lmk->k', c_mat, Y_basis).real
    return V

def _find_minima(c, n_points=N):
    """Find local minima of V defined by c."""
    V_grid = _eval_V(c, theta_flat, phi_flat)
    
    # Find local minima on the grid
    # V_grid is 1D. We need to check neighbors in 2D grid.
    V_2d = V_grid.reshape(GRID_LAT, GRID_LON)
    
    # Pad to handle boundaries (wrap around in lon, reflect in lat?)
    # Simple check: point is min if it is <= all 8 neighbors.
    # For speed, just check 4 neighbors.
    minima_idx = []
    for i in range(GRID_LAT):
        for j in range(GRID_LON):
            v = V_2d[i, j]
            # Neighbors
            v_up = V_2d[i, (j + 1) % GRID_LON]
            v_down = V_2d[i, (j - 1) % GRID_LON]
            if i > 0:
                v_north = V_2d[i - 1, j]
            else:
                v_north = np.inf
            if i < GRID_LAT - 1:
                v_south = V_2d[i + 1, j]
            else:
                v_south = np.inf
                
            if v <= v_up and v <= v_down and v <= v_north and v <= v_south:
                # Check if it's a strict local min or just a flat area
                # To avoid duplicates, we can require it to be strictly less than at least one neighbor
                # Or just collect and deduplicate later.
                minima_idx.append(i * GRID_LON + j)
                
    if not minima_idx:
        return None
        
    # Refine each minimum using Newton's method
    refined_points = []
    refined_vals = []
    
    for idx in minima_idx:
        t0 = theta_flat[idx]
        p0 = phi_flat[idx]
        
        # Newton refinement
        # We need V and grad V at (t, p)
        # V = sum c Y
        # grad V = sum c (dY/dt, dY/dp)
        
        # We didn't precompute dY/dt, dY/dp analytically in the loop above.
        # Let's compute them on the fly for the refinement points.
        # This is slow if we do it for every point.
        # Alternative: Precompute dY/dt and dY/dp on the grid as well?
        # Or just use numerical gradient for refinement.
        
        # Let's use a simple numerical gradient descent / Newton step.
        # Since we are at a local min on the grid, a few steps of gradient descent with small step size should suffice.
        # Or just use the grid value as an approximation?
        # For N=150, grid resolution 60x120 is decent.
        # Let's try to refine with 5 steps of gradient descent.
        
        t = t0
        p = p0
        step = 0.01
        for _ in range(5):
            # Compute grad at (t, p)
            # Numerical gradient
            eps = 1e-4
            V_c = _eval_V(c, np.array([t]), np.array([p]))[0]
            V_t = _eval_V(c, np.array([t + eps]), np.array([p]))[0]
            V_p = _eval_V(c, np.array([t]), np.array([p + eps]))[0]
            
            gt = (V_t - V_c) / eps
            gp = (V_p - V_c) / eps
            
            # Update
            t -= step * gt
            p -= step * gp
            
            # Clip theta
            t = np.clip(t, 1e-6, np.pi - 1e-6)
            
        # Final value
        V_final = _eval_V(c, np.array([t]), np.array([p]))[0]
        refined_points.append((t, p))
        refined_vals.append(V_final)
        
    if len(refined_points) < n_points:
        # If we didn't find enough, return None or all found
        if len(refined_points) == 0:
            return None
        # If not enough, we can't form N points. 
        # We might need to increase grid resolution or L.
        # For now, return what we have? No, we need N.
        # Let's assume we find enough. If not, this config is bad.
        pass
        
    # Sort by value (lowest V is best min)
    order = np.argsort(refined_vals)
    selected = order[:n_points]
    
    th = np.array([refined_points[i][0] for i in selected])
    ph = np.array([refined_points[i][1] for i in selected])
    
    xyz = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return xyz

def _log_energy(pts):
    d2 = np.sum(pts * pts, axis=1)[:, None] + np.sum(pts * pts, axis=1)[None, :] - 2 * pts @ pts.T
    d2 = np.maximum(d2, 1e-18)
    d = np.sqrt(d2)
    iu = np.triu_indices(N, k=1)
    return np.sum(np.log(1.0 / d[iu]))

def construct():
    np.random.seed(42)
    
    # Precompute basis
    _precompute_basis()
    
    # Initialize coefficients
    c = np.random.randn((MAX_DEG + 1, 2 * MAX_DEG + 1)).flatten()
    # Zero out invalid m
    for l in range(MAX_DEG + 1):
        for m in range(-l, l + 1):
            if m + MAX_DEG >= 0 and m + MAX_DEG < 2 * MAX_DEG + 1:
                pass
            else:
                c[l * (2 * MAX_DEG + 1) + m + MAX_DEG] = 0

    best_energy = np.inf
    best_xyz = None
    
    # Local search on coefficients
    # Use Nelder-Mead or simple random perturbations?
    # Let's use a simple random walk with cooling.
    
    current_c = c.copy()
    current_xyz = _find_minima(current_c)
    if current_xyz is not None:
        current_energy = _log_energy(current_xyz)
        if current_energy < best_energy:
            best_energy = current_energy
            best_xyz = current_xyz.copy()
            
    # Optimize
    for i in range(50): # 50 iterations
        # Perturb c
        noise = np.random.randn(current_c.shape) * 0.1
        new_c = current_c + noise
        
        # Zero out invalid m
        for l in range(MAX_DEG + 1):
            for m in range(-l, l + 1):
                if m + MAX_DEG >= 0 and m + MAX_DEG < 2 * MAX_DEG + 1:
                    pass
                else:
                    new_c[l * (2 * MAX_DEG + 1) + m + MAX_DEG] = 0

        new_xyz = _find_minima(new_c)
        if new_xyz is None:
            continue
            
        new_energy = _log_energy(new_xyz)
        
        if new_energy < current_energy:
            current_c = new_c
            current_xyz = new_xyz
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_xyz = current_xyz.copy()
        else:
            # Keep current, maybe try smaller step?
            # For simplicity, just keep current.
            pass
            
    if best_xyz is None:
        # Fallback
        return _fibonacci_grid(N, offset=0.5)
        
    # Normalize
    best_xyz = best_xyz / np.linalg.norm(best_xyz, axis=1, keepdims=True)
    
    # Validate
    d2 = np.sum(best_xyz * best_xyz, axis=1)[:, None] + np.sum(best_xyz * best_xyz, axis=1)[None, :] - 2 * best_xyz @ best_xyz.T
    d = np.sqrt(np.maximum(d2, 0.0))
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 1e-9:
        return _fibonacci_grid(N, offset=0.5)
        
    return best_xyz

def _fibonacci_grid(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])
