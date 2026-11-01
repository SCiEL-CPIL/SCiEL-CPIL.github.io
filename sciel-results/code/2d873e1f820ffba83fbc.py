import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 14
D = 3

# Precompute pair indices for efficiency
_pair_indices = list(itertools.combinations(range(N), 2))
M = len(_pair_indices)
_pair_i = np.array([p[0] for p in _pair_indices])
_pair_j = np.array([p[1] for p in _pair_indices])

def _compute_distances_sq(X):
    """
    Compute squared pairwise distances.
    X: (N, D)
    Returns: (M,) array of squared distances.
    """
    diff = X[_pair_i] - X[_pair_j]
    dist_sq = np.sum(diff**2, axis=1)
    return dist_sq

def _objective_and_grad(x_flat, tau):
    """
    x_flat: (N*D,)
    tau: temperature
    Returns: objective value, gradient
    """
    X = x_flat.reshape(N, D)
    # Center of mass
    X = X - np.mean(X, axis=0, keepdims=True)
    
    dist_sq = _compute_distances_sq(X)
    
    # Soft max
    max_val = np.max(dist_sq)
    # Avoid underflow/overflow
    exp_max = np.exp(tau * (dist_sq - max_val))
    Z_max = np.sum(exp_max)
    S_max = max_val + np.log(Z_max) / tau
    p_max = exp_max / Z_max
    
    # Soft min
    min_val = np.min(dist_sq)
    exp_min = np.exp(-tau * (dist_sq - min_val))
    Z_min = np.sum(exp_min)
    S_min = min_val - np.log(Z_min) / tau
    p_min = exp_min / Z_min
    
    if S_min <= 0 or S_max <= 0:
        return 1e10, np.zeros_like(x_flat)
        
    obj = np.log(S_max) - np.log(S_min)
    
    grad_S_max = np.zeros_like(X)
    grad_S_min = np.zeros_like(X)
    
    diff = X[_pair_i] - X[_pair_j]
    
    # Gradient of S_max = sum p_k * d_k^2
    # d(S_max)/dX_i = sum_k p_k * d(d_k^2)/dX_i
    # d(d_k^2)/dX_i = 2*(X_i - X_j) if i is in pair, -2*(X_i - X_j) if j is in pair
    np.add.at(grad_S_max, _pair_i, 2.0 * p_max[:, np.newaxis] * diff)
    np.add.at(grad_S_max, _pair_j, -2.0 * p_max[:, np.newaxis] * diff)
    
    # Gradient of S_min = sum p_k * d_k^2 (where p_k are weights for min)
    # Note: S_min is a soft minimum, so it behaves like a weighted average of the smallest distances.
    # The derivative is similar structure.
    np.add.at(grad_S_min, _pair_i, 2.0 * p_min[:, np.newaxis] * diff)
    np.add.at(grad_S_min, _pair_j, -2.0 * p_min[:, np.newaxis] * diff)
    
    # Objective is log(S_max) - log(S_min)
    # Grad = (1/S_max) * grad_S_max - (1/S_min) * grad_S_min
    grad_J = grad_S_max / S_max - grad_S_min / S_min
    
    return obj, grad_J.flatten()

def _objective_qcqp(x_flat, t_target, lam=100.0, tau_min=100.0):
    """
    QCQP-style objective: maximize min distance subject to max distance <= t_target.
    Minimizes -log(soft_min) + penalty for exceeding t_target.
    """
    X = x_flat.reshape(N, D)
    X = X - np.mean(X, axis=0, keepdims=True)
    
    dist_sq = _compute_distances_sq(X)
    
    # Soft min
    min_val = np.min(dist_sq)
    exp_min = np.exp(-tau_min * (dist_sq - min_val))
    Z_min = np.sum(exp_min)
    S_min = min_val - np.log(Z_min) / tau_min
    p_min = exp_min / Z_min
    
    if S_min <= 0:
        return 1e10, np.zeros_like(x_flat)
        
    # Penalty for exceeding t_target
    viol = np.maximum(dist_sq - t_target, 0.0)
    penalty = lam * np.sum(viol**2)
    
    obj = -np.log(S_min) + penalty
    
    # Gradient
    grad_S_min = np.zeros_like(X)
    diff = X[_pair_i] - X[_pair_j]
    np.add.at(grad_S_min, _pair_i, 2.0 * p_min[:, np.newaxis] * diff)
    np.add.at(grad_S_min, _pair_j, -2.0 * p_min[:, np.newaxis] * diff)
    
    grad_penalty = np.zeros_like(X)
    # d/dX sum max(d^2 - t, 0)^2
    # Only active where d^2 > t
    active = dist_sq > t_target
    if np.any(active):
        # grad of (d^2 - t)^2 is 2(d^2 - t) * grad(d^2)
        coeff = 2.0 * (dist_sq[active] - t_target)
        # Add to gradient
        np.add.at(grad_penalty, _pair_i[active], 2.0 * coeff[:, np.newaxis] * diff[active])
        np.add.at(grad_penalty, _pair_j[active], -2.0 * coeff[:, np.newaxis] * diff[active])
        
    grad_J = -grad_S_min / S_min + lam * grad_penalty
    
    return obj, grad_J.flatten()

def _exact_score(X):
    dist_sq = _compute_distances_sq(X)
    d_min = np.min(dist_sq)
    d_max = np.max(dist_sq)
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min)

def _fibonacci_sphere(n, d):
    if d == 3:
        points = np.zeros((n, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            z = 1.0 - (i / float(n - 1)) * 2.0
            radius = np.sqrt(1.0 - z * z)
            theta = golden_angle * i
            points[i, 0] = np.cos(theta) * radius
            points[i, 1] = np.sin(theta) * radius
            points[i, 2] = z
        return points
    else:
        points = np.random.randn(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin for the 150s limit

    best_score = np.inf
    best_X = None
    
    num_restarts = 100 # Higher limit, break on time
    
    rng = np.random.default_rng(42)
    
    # Pre-generate some structured initializations
    # 1. Fibonacci sphere
    fib_pts = _fibonacci_sphere(N, D)
    
    # 2. Icosahedron + 2 points (poles)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    ico_plus = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    
    # 3. Tetrahedron + midpoints + face centers
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    edge_midpoints = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (tetra[i] + tetra[j]) / 2
            mid = mid / np.linalg.norm(mid)
            edge_midpoints.append(mid)
            
    face_centers = []
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                center = (tetra[i] + tetra[j] + tetra[k]) / 3
                center = center / np.linalg.norm(center)
                face_centers.append(center)
                
    tetra_full = np.vstack([tetra, np.array(edge_midpoints), np.array(face_centers[:4])])
    
    structured_inits = [fib_pts, ico_plus, tetra_full]
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0: # Leave 10s for final polish
            break
            
        # Generate initial points
        if restart < len(structured_inits):
            X = structured_inits[restart].copy()
            # Add small noise to break symmetry if needed
            X += rng.normal(scale=0.01, size=(N, D))
        elif restart % 3 == 0:
            # Fibonacci Sphere with noise
            X = _fibonacci_sphere(N, D)
            X += rng.normal(scale=0.1, size=(N, D))
        elif restart % 3 == 1:
            # Random on sphere
            X = rng.normal(size=(N, D))
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / norms
        else:
            # Random in unit ball
            X = rng.normal(size=(N, D))
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / norms * rng.uniform(0.1, 1.0, size=(N, 1))
            
        # Center
        X -= np.mean(X, axis=0, keepdims=True)
        
        # Normalize scale to avoid numerical issues
        # Set mean squared distance to 1
        dist_sq = _compute_distances_sq(X)
        mean_dist_sq = np.mean(dist_sq)
        if mean_dist_sq > 0:
            X = X * np.sqrt(1.0 / mean_dist_sq)
            
        x_flat = X.flatten()
        
        # Phase 1: Progressive tau smooth optimization
        taus = [5, 20, 100, 500, 2000]
        
        for tau in taus:
            if time.time() > deadline - 10.0:
                break
                
            try:
                res = minimize(
                    _objective_and_grad, 
                    x_flat, 
                    args=(tau,), 
                    jac=True, 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.x is not None:
                    x_flat = res.x
            except Exception:
                pass
        
        # Phase 2: QCQP refinement
        if time.time() < deadline - 10.0:
            X_current = x_flat.reshape(N, D)
            X_current -= np.mean(X_current, axis=0, keepdims=True)
            dist_sq = _compute_distances_sq(X_current)
            t_target = np.max(dist_sq)
            
            try:
                res_qcqp = minimize(
                    _objective_qcqp, 
                    x_flat, 
                    args=(t_target, 100.0, 100.0), 
                    jac=True, 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_qcqp.x is not None:
                    x_flat = res_qcqp.x
            except Exception:
                pass
            
            # Second QCQP pass with updated t_target
            if time.time() < deadline - 10.0:
                X_current = x_flat.reshape(N, D)
                X_current -= np.mean(X_current, axis=0, keepdims=True)
                dist_sq = _compute_distances_sq(X_current)
                t_target = np.max(dist_sq)
                
                try:
                    res_qcqp2 = minimize(
                        _objective_qcqp, 
                        x_flat, 
                        args=(t_target, 100.0, 100.0), 
                        jac=True, 
                        method='L-BFGS-B', 
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res_qcqp2.x is not None:
                        x_flat = res_qcqp2.x
                except Exception:
                    pass
        
        # Phase 3: Nelder-Mead polish on exact objective
        def nm_obj(x_flat):
            X = x_flat.reshape(N, D)
            X = X - np.mean(X, axis=0, keepdims=True)
            return _exact_score(X)
            
        if time.time() < deadline - 5.0:
            try:
                res_nm = minimize(
                    nm_obj, 
                    x_flat, 
                    method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_nm.x is not None:
                    x_flat = res_nm.x
            except Exception:
                pass
        
        # Final evaluation
        X_final = x_flat.reshape(N, D)
        X_final -= np.mean(X_final, axis=0, keepdims=True)
        
        dist_sq = _compute_distances_sq(X_final)
        if np.min(dist_sq) < 1e-8:
            continue
            
        score = _exact_score(X_final)
        
        if score < best_score:
            best_score = score
            best_X = X_final.copy()
            
    if best_X is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_X = np.array(grid, dtype=float)
        
    # Final high-precision polish on the best point found
    if best_X is not None and time.time() < deadline - 2.0:
        x_best = best_X.flatten()
        
        # Try one more SLSQP pass with direct objective
        def direct_obj(x_flat):
            X = x_flat.reshape(N, D)
            X = X - np.mean(X, axis=0, keepdims=True)
            return _exact_score(X)
            
        try:
            res_slsqp = minimize(
                direct_obj, 
                x_best, 
                method='SLSQP', 
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            if res_slsqp.x is not None:
                X_check = res_slsqp.x.reshape(N, D)
                X_check -= np.mean(X_check, axis=0, keepdims=True)
                score_check = _exact_score(X_check)
                if score_check < best_score:
                    best_score = score_check
                    best_X = X_check.copy()
        except Exception:
            pass
        
        # Try one more L-BFGS-B with very high tau
        if time.time() < deadline - 1.0:
            try:
                res_final = minimize(
                    _objective_and_grad, 
                    best_X.flatten(), 
                    args=(10000.0,), 
                    jac=True, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
                )
                if res_final.x is not None:
                    X_check = res_final.x.reshape(N, D)
                    X_check -= np.mean(X_check, axis=0, keepdims=True)
                    score_check = _exact_score(X_check)
                    if score_check < best_score:
                        best_score = score_check
                        best_X = X_check.copy()
            except Exception:
                pass

    return best_X

if __name__ == "__main__":
    pts = construct()
    score = _exact_score(pts)
    print(f"Score: {score}")
