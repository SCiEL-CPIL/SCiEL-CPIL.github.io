import numpy as np
import scipy.optimize as opt
import time

N = 14
DEADLINE = 140  # 10s safety margin before 150s kill

def _triangle_areas_from_angles(angles):
    """
    Compute all C(14,3) triangle areas for points on unit circle at given angles.
    Points are (cos(a), sin(a)) for angle a.
    Area of triangle with vertices at angles a,b,c on unit circle:
    = 0.5 * |sin(b-a) + sin(c-b) + sin(a-c)|
    """
    n = len(angles)
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ai = angles[i]
                aj = angles[j]
                ak = angles[k]
                # Area = 0.5 * |sin(aj-ai) + sin(ak-aj) + sin(ai-ak)|
                area = 0.5 * abs(np.sin(aj - ai) + np.sin(ak - aj) + np.sin(ai - ak))
                areas[i, j, k] = area
    return areas

def _min_triangle_area(angles, tau=0.05):
    """
    Soft-min of all triangle areas using log-sum-exp trick.
    Returns (soft_min_value, gradient_wrt_angles)
    """
    n = len(angles)
    # Compute all triangle areas vectorized
    # For unit circle points, area(i,j,k) = 0.5*|sin(j-i)+sin(k-j)+sin(i-k)|
    # We need to handle the abs() carefully for gradients.
    
    # Precompute sin differences
    # For efficiency, compute all pairs
    areas_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ai = angles[i]
                aj = angles[j]
                ak = angles[k]
                s = np.sin(aj - ai) + np.sin(ak - aj) + np.sin(ai - ak)
                area = 0.5 * abs(s)
                areas_list.append(area)
    
    areas_arr = np.array(areas_list)
    
    # Soft min: -tau * log(sum(exp(-area/tau)))
    # For numerical stability
    min_a = np.min(areas_arr)
    exp_terms = np.exp(-(areas_arr - min_a) / tau)
    soft_min = min_a - tau * np.log(np.sum(exp_terms))
    
    # Gradient: d(soft_min)/d(angle_m)
    # soft_min = min_a - tau * log(sum(exp(-(a_m - min_a)/tau)))
    # Let w_k = exp(-(a_k - min_a)/tau) / sum(exp(...))
    # d(soft_min)/d(a_m) = -sum_k w_k * (1/tau) * d(a_k)/d(a_m) ... 
    # Actually: soft_min = min_a - tau*log(S) where S = sum exp(-(a_k-min_a)/tau)
    # d(soft_min)/d(x) = d(min_a)/dx - (1/S)*dS/dx
    # dS/dx = sum_k exp(-(a_k-min_a)/tau) * (-(1/tau)*da_k/dx + (1/tau)*dmin_a/dx)
    # This is complex. Let's use a simpler formulation.
    
    # Alternative: soft_min = -tau * log(sum(exp(-a_k/tau)))
    # d(soft_min)/dx = sum_k [exp(-a_k/tau)/sum(exp(-a_j/tau))] * (1/tau) * da_k/dx
    # = (1/tau) * sum_k w_k * da_k/dx
    # where w_k = exp(-a_k/tau) / sum_j exp(-a_j/tau)
    
    # For numerical stability, shift by min
    shifted = areas_arr - min_a
    exp_shifted = np.exp(-shifted / tau)
    S = np.sum(exp_shifted)
    weights = exp_shifted / S  # w_k
    
    # Now compute da_k/d(angle_m) for each triangle k and each angle m
    # Triangle k corresponds to indices (i,j,k)
    # a_k = 0.5 * |sin(aj-ai) + sin(ak-aj) + sin(ai-ak)|
    # da/d(ai) = 0.5 * sign(s) * (-cos(aj-ai) + cos(ai-ak))  [derivative of sin(aj-ai) wrt ai is -cos(aj-ai), derivative of sin(ai-ak) wrt ai is cos(ai-ak)]
    # Wait: s = sin(aj-ai) + sin(ak-aj) + sin(ai-ak)
    # ds/dai = -cos(aj-ai) + cos(ai-ak)
    # ds/daj = cos(aj-ai) - cos(ak-aj)
    # ds/dak = -cos(ak-aj) + cos(ai-ak)
    # da/dam = 0.5 * sign(s) * ds/dam
    
    # Build mapping: triangle index -> (i,j,k)
    tri_indices = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_indices.append((i, j, k))
                idx += 1
    
    grad = np.zeros(n)
    for t_idx, (i, j, k) in enumerate(tri_indices):
        w = weights[t_idx]
        if w < 1e-12:
            continue
        ai = angles[i]
        aj = angles[j]
        ak = angles[k]
        s = np.sin(aj - ai) + np.sin(ak - aj) + np.sin(ai - ak)
        sign_s = 1.0 if s >= 0 else -1.0
        
        # ds/dai
        ds_dai = -np.cos(aj - ai) + np.cos(ai - ak)
        # ds/daj
        ds_daj = np.cos(aj - ai) - np.cos(ak - aj)
        # ds/dak
        ds_dak = -np.cos(ak - aj) + np.cos(ai - ak)
        
        coeff = 0.5 * sign_s * w / tau
        grad[i] += coeff * ds_dai
        grad[j] += coeff * ds_daj
        grad[k] += coeff * ds_dak
    
    return soft_min, grad

def _neg_soft_min(angles, tau=0.05):
    """Negative soft-min for minimization."""
    val, _ = _min_triangle_area(angles, tau)
    return -val

def _neg_soft_min_grad(angles, tau=0.05):
    """Negative soft-min with gradient for L-BFGS-B."""
    val, grad = _min_triangle_area(angles, tau)
    return -val, -grad

def _score_from_angles(angles):
    """
    Score = min triangle area / hull area.
    For points on unit circle, hull is the regular-ish polygon.
    Hull area for points on unit circle in angular order:
    A_hull = 0.5 * sum_{i} |cross(P_i, P_{i+1})|
           = 0.5 * sum_{i} |cos(a_i)sin(a_{i+1}) - sin(a_i)cos(a_{i+1})|
           = 0.5 * sum_{i} |sin(a_{i+1} - a_i)|
    """
    n = len(angles)
    # Sort angles to ensure proper ordering
    sorted_angles = np.sort(angles)
    
    # Min triangle area
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ai = sorted_angles[i]
                aj = sorted_angles[j]
                ak = sorted_angles[k]
                area = 0.5 * abs(np.sin(aj - ai) + np.sin(ak - aj) + np.sin(ai - ak))
                if area < min_area:
                    min_area = area
    
    # Hull area
    hull_area = 0.0
    for i in range(n):
        a_i = sorted_angles[i]
        a_next = sorted_angles[(i+1) % n]
        diff = a_next - a_i
        if diff < 0:
            diff += 2 * np.pi
        hull_area += 0.5 * abs(np.sin(diff))
    
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Initial angle configurations
    initial_configs = []
    
    # Regular polygon angles
    regular = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_configs.append(regular.copy())
    
    # Regular polygon with small perturbation
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbed = regular + rng.normal(0, 0.05, N)
        initial_configs.append(perturbed)
    
    # Slightly non-uniform: group some closer
    for _ in range(3):
        angles = np.sort(rng.uniform(0, 2*np.pi, N))
        initial_configs.append(angles)
    
    # Two-cluster configuration
    for _ in range(2):
        half = N // 2
        a1 = np.linspace(0, np.pi, half, endpoint=False) + rng.uniform(0, 0.1)
        a2 = np.linspace(np.pi, 2*np.pi, N-half, endpoint=False) + rng.uniform(0, 0.1)
        angles = np.sort(np.concatenate([a1, a2]))
        initial_configs.append(angles)
    
    for init_angles in initial_configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        
        # Normalize angles to [0, 2*pi)
        init_angles = np.sort(init_angles % (2*np.pi))
        
        # Anneal tau from large to small
        tau_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        
        current_angles = init_angles.copy()
        
        for tau in tau_values:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
            
            try:
                result = opt.minimize(
                    _neg_soft_min_grad,
                    current_angles,
                    args=(tau,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
                )
                if result.fun is not None:
                    current_angles = result.x
            except Exception:
                pass
        
        # Final polish with very small tau
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 5:
            try:
                result = opt.minimize(
                    _neg_soft_min_grad,
                    current_angles,
                    args=(0.005,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
                )
                if result.fun is not None:
                    current_angles = result.x
            except Exception:
                pass
        
        # Convert angles to points and compute score
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 2:
            break
        
        angles = np.sort(current_angles % (2*np.pi))
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        score = _score_from_angles(angles)
        if score > best_score:
            best_score = score
            best_points = points
    
    # If we didn't find anything, use regular polygon
    if best_points is None:
        regular = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(regular), np.sin(regular)])
    
    return np.array(best_points)
