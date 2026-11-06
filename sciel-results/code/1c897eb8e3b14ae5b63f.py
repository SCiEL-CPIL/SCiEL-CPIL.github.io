import numpy as np
import time
import sys
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius in radians.
    The covering radius is the maximum of the circumradii of the facets of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_circumradius = 0.0
        for i, j, k in facets:
            p_i = points[i]
            p_j = points[j]
            p_k = points[k]
            
            c = p_i + p_j + p_k
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                continue
            c = c / norm_c
            
            dot = np.dot(c, p_i)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)
            
            if theta > max_circumradius:
                max_circumradius = theta
                
        return max_circumradius
    except Exception:
        return np.pi

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _spectral_energy(points, l_max=20):
    """
    Computes a spectral energy proxy for uniformity.
    Uses the variance of the pair correlation function.
    For a uniform distribution on the sphere, the spherical harmonic coefficients 
    (other than l=0) should be close to zero.
    We minimize the sum of squares of the coefficients for l > 0.
    """
    # Compute all pairwise dot products (cosines of angular distances)
    # This is equivalent to the kernel matrix of the spherical harmonics up to a certain degree
    # Specifically, the sum of squares of the spherical harmonic coefficients of the 
    # empirical measure is related to the integral of the square of the kernel.
    
    # A simpler and more robust spectral proxy:
    # Sum_{l=1}^{l_max} |C_l|^2 where C_l are the coefficients of the power spectrum.
    # Actually, the "variance of the pair correlation" is often approximated by 
    # minimizing the sum of squared spherical harmonic coefficients of the point cloud.
    
    # Let's use the sum of squares of the components of the points projected onto 
    # spherical harmonics. 
    # For efficiency and simplicity, we can use the fact that:
    # Sum_{i,j} K(x_i, x_j) where K is the kernel.
    # The kernel for spherical harmonics up to degree L is related to the Legendre polynomials.
    # P_l(cos(theta)) = sum_{m=-l}^l Y_{lm}(x) Y_{lm}(y)^*
    
    # A common energy function is the Riesz s-energy or logarithmic energy.
    # However, the prompt asks for "variance of the pair correlation function".
    # The pair correlation function g(r) is related to the power spectrum S(l).
    # Minimizing the variance of g(r) is equivalent to minimizing the sum of S(l) for l>0.
    
    # Let's compute the power spectrum.
    # C_l = (1/N) sum_i Y_l(x_i) -- wait, this is for the mean.
    # The power spectrum of the point process is |sum_i Y_l(x_i)|^2.
    
    # We need to evaluate spherical harmonics. This is expensive if done from scratch.
    # Alternative: Use the fact that for a set of points, the sum of squared 
    # spherical harmonic coefficients is:
    # E = sum_{l=1}^{l_max} (2l+1) * | (1/N) sum_{i=1}^N Y_l(x_i) |^2 ? No.
    
    # Let's use a simpler proxy that is known to promote uniformity:
    # The sum of pairwise squared distances is constant for points on a sphere.
    # The sum of pairwise 4th power distances?
    
    # Let's stick to the spectral idea but implement it efficiently.
    # The "spectral energy" can be defined as:
    # E = sum_{l=1}^{l_max} |c_l|^2 where c_l are the coefficients of the 
    # spherical harmonic expansion of the delta comb.
    
    # Actually, a very effective and simple spectral-like energy is:
    # Minimize the sum of the squared norms of the projections of the points 
    # onto each spherical harmonic subspace.
    
    # For implementation simplicity and speed within 150s, let's use a 
    # gradient-based optimization on a smooth energy function that correlates 
    # with uniformity.
    
    # The "variance of the pair correlation" is minimized when the points are 
    # as uniform as possible.
    
    # Let's use the following energy:
    # E = sum_{i != j} f(d_ij) where f is a function that penalizes clustering.
    # But the prompt specifically asks for a spectral approach.
    
    # We can approximate the spectral energy using the trace of the kernel matrix.
    # K = sum_{l=0}^{l_max} (2l+1) P_l(X X^T)
    # The energy is proportional to sum_{i != j} K_{ij}.
    
    # Let's compute the pairwise dot products.
    dots = points @ points.T
    # Ensure diagonal is 1
    np.fill_diagonal(dots, 1.0)
    
    # We want to minimize the off-diagonal sum of the kernel.
    # For l=1, the kernel is related to the dot product.
    # For higher l, we need Legendre polynomials.
    
    # Let's sum up to l_max=10 for stability and speed.
    energy = 0.0
    for l in range(1, l_max + 1):
        # Compute P_l(dots)
        # We can use recurrence relation for Legendre polynomials
        if l == 1:
            p_l = dots
        else:
            # Need p_{l-1} and p_{l-2}
            # This requires storing previous polynomials
            pass
    
    # Recurrence approach:
    p_0 = np.ones_like(dots)
    p_1 = dots
    energy = 0.0
    # l=1 term: (2*1+1) * sum(p_1) - N (remove diagonal)
    # Actually, we want sum_{i != j} P_l(d_ij)
    
    for l in range(1, l_max + 1):
        if l == 1:
            p_l = p_1
        elif l == 2:
            p_l = (3 * dots * p_1 - p_0) / 2.0
        else:
            # We need to keep track of previous two
            pass
            
    # Let's just use l=1 and l=2 for simplicity and speed, or use a library if available.
    # SciPy has lpmv for associated Legendre functions, but standard Legendre is easy.
    
    # Let's implement the recurrence properly.
    p_lm2 = p_0
    p_lm1 = p_1
    energy = 0.0
    
    for l in range(1, l_max + 1):
        if l == 1:
            p_l = p_lm1
        else:
            p_l = ((2 * l - 1) * dots * p_lm1 - (l - 1) * p_lm2) / l
            p_lm2 = p_lm1
            p_lm1 = p_l
            
        # Sum of off-diagonal elements
        sum_off_diag = np.sum(p_l) - np.trace(p_l)
        energy += (2 * l + 1) * sum_off_diag
        
    return energy

def _spectral_gradient(points, l_max=20):
    """
    Compute gradient of the spectral energy.
    This is complex to derive analytically for the sum of Legendre polynomials.
    We can use finite differences or derive the gradient of the kernel sum.
    
    For K(x, y) = sum_{l} (2l+1) P_l(x.y)
    d/dx_i K(x_i, x_j) = sum_{l} (2l+1) P_l'(x_i.x_j) * x_j
    
    P_l'(t) = l * (P_{l-1}(t) - t P_l(t)) / (1 - t^2)  -- for t != +/- 1
    Or use recurrence.
    
    Given the complexity and time limit, we might use a simple gradient descent 
    with finite differences or a more robust optimizer like L-BFGS-B 
    which uses finite differences if gradient is not provided, or we can 
    provide an approximate gradient.
    
    However, L-BFGS-B with finite differences for 130*3=390 variables 
    and a function that takes ~0.1s to evaluate might be too slow 
    (390 evaluations per step, ~40s per step).
    
    Let's try to compute the gradient analytically or semi-analytically.
    
    Actually, let's use a simpler energy that has a known gradient:
    The sum of pairwise inverse distances (Riesz s=1 energy) or similar.
    But the prompt asks for spectral.
    
    Let's try to use the fact that the gradient of the sum of P_l(d_ij) 
    can be computed efficiently if we precompute the Legendre polynomials and 
    their derivatives.
    
    Alternatively, we can use a different spectral proxy:
    Minimize the variance of the local density.
    
    Given the constraints, let's use a hybrid:
    1. Use a smooth energy function that is easy to differentiate.
    2. The "spectral" aspect can be captured by using a kernel that is a 
       sum of Legendre polynomials, but we might simplify the gradient 
       computation or use a lower l_max.
       
    Let's use l_max=5 for the gradient computation to keep it fast.
    """
    dots = points @ points.T
    np.fill_diagonal(dots, 1.0)
    
    grad = np.zeros_like(points)
    
    # Precompute P_l and P_l' for l=1..l_max
    # P_0 = 1, P_1 = t
    # P_0' = 0, P_1' = 1
    
    p_vals = [None] * (l_max + 1)
    dp_vals = [None] * (l_max + 1)
    
    p_vals[0] = np.ones_like(dots)
    dp_vals[0] = np.zeros_like(dots)
    
    p_vals[1] = dots
    dp_vals[1] = np.ones_like(dots)
    
    for l in range(2, l_max + 1):
        p_vals[l] = ((2 * l - 1) * dots * p_vals[l-1] - (l - 1) * p_vals[l-2]) / l
        # Derivative:
        # P_l'(t) = l * (P_{l-1}(t) - t P_l(t)) / (1 - t^2)
        # This is unstable near t=1.
        # Alternative: P_l'(t) = (l * P_{l-1}(t) - l * t * P_l(t)) / (1 - t^2)
        # Or use: P_l'(t) = (2l-1)(P_{l-1}(t) - t P_l(t))/(1-t^2) ? No.
        # Standard identity: (1-t^2) P_l'(t) = l (P_{l-1}(t) - t P_l(t))
        # So P_l'(t) = l (P_{l-1}(t) - t P_l(t)) / (1 - t^2)
        
        # To avoid division by zero, we can use a small epsilon or 
        # use the recurrence for derivatives directly.
        # P_l'(t) = (2l-1) P_{l-1}(t) - t P_l'(t) ? No.
        # P_l'(t) = l P_{l-1}(t) - l t P_l(t) / (1-t^2)
        
        # Let's use the stable recurrence:
        # P_l'(t) = (2l-1) P_{l-1}(t) - t P_l'(t) is not correct.
        # Correct: P_l'(t) = l * (P_{l-1}(t) - t P_l(t)) / (1 - t^2)
        
        # For numerical stability, we can compute the gradient using 
        # automatic differentiation or finite differences for the energy, 
        # but that's slow.
        
        # Let's simplify: Only use l=1 and l=2 for the gradient.
        # This is a "low-order spectral" approximation.
        break
        
    # Gradient for l=1:
    # d/dx_i sum_{j != i} P_1(x_i.x_j) = d/dx_i sum_{j != i} (x_i.x_j) = sum_{j != i} x_j
    # Note: sum_{j} x_j includes x_i, so sum_{j != i} x_j = sum_{j} x_j - x_i
    sum_points = np.sum(points, axis=0)
    grad_l1 = sum_points[None, :] - points
    
    # Gradient for l=2:
    # P_2(t) = (3t^2 - 1)/2
    # P_2'(t) = 3t
    # d/dx_i sum_{j != i} P_2(x_i.x_j) = sum_{j != i} P_2'(x_i.x_j) * x_j
    # = sum_{j != i} 3 (x_i.x_j) x_j
    # = 3 (x_i^T X^T x_i - x_i^T x_i) ... wait.
    # sum_{j} (x_i.x_j) x_j = X^T (X x_i) = X^T X x_i
    # sum_{j != i} (x_i.x_j) x_j = X^T X x_i - (x_i.x_i) x_i = X^T X x_i - x_i
    # So grad_l2 = 3 * (X^T X points - points)
    
    XtX = points.T @ points
    grad_l2 = 3.0 * (XtX @ points - points)
    
    # Combine: Energy = 3 * sum_off_diag(P_1) + 5 * sum_off_diag(P_2) + ...
    # We used coefficients (2l+1).
    # So gradient = 3 * grad_l1 + 5 * grad_l2
    
    grad = 3.0 * grad_l1 + 5.0 * grad_l2
    
    return grad

def construct():
    """
    Constructs a set of N points on the unit sphere to minimize the covering radius.
    Strategy: Spectral energy minimization (low-order) followed by local geometric refinement.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin

    # 1. Initialization: Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # 2. Spectral Optimization Phase
    # Use L-BFGS-B with the analytical gradient (low-order approximation)
    # We optimize in Cartesian coordinates and project back to sphere?
    # No, L-BFGS-B works in Euclidean space. We need to constrain to sphere.
    # We can use the "spherical gradient" by projecting the Euclidean gradient 
    # onto the tangent plane at each point.
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
        
        # Compute energy
        # For speed, we use the low-order energy directly
        dots = x @ x.T
        np.fill_diagonal(dots, 1.0)
        
        # Energy = 3 * (sum(dots) - N) + 5 * (sum(P_2(dots)) - N)
        # P_2(t) = (3t^2 - 1)/2
        p2 = (3 * dots**2 - 1) / 2.0
        energy = 3.0 * (np.sum(dots) - N) + 5.0 * (np.sum(p2) - N)
        
        # Gradient
        grad = _spectral_gradient(x, l_max=2)
        
        # Project gradient to tangent plane
        # grad_tangent = grad - (grad . x) x
        grad_proj = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        
        return energy, grad_proj.flatten()
    
    # Initial point
    x0 = points.flatten()
    
    # Run optimization
    try:
        res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', 
                       options={'maxiter': 50, 'gtol': 1e-8, 'maxls': 20})
        points = res.x.reshape(N, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > time_limit * 0.5:
        return points

    # 3. Local Geometric Refinement Phase
    # Use the current points as a starting point for local search on the 
    # actual covering radius.
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    rng = np.random.default_rng(42)
    
    def get_worst_facet(pts):
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            max_cr = 0.0
            worst_idx = -1
            for idx, (i, j, k) in enumerate(facets):
                p_i = pts[i]
                p_j = pts[j]
                p_k = pts[k]
                c = p_i + p_j + p_k
                norm_c = np.linalg.norm(c)
                if norm_c < 1e-10:
                    continue
                c = c / norm_c
                dot = np.clip(np.dot(c, p_i), -1.0, 1.0)
                theta = np.arccos(dot)
                if theta > max_cr:
                    max_cr = theta
                    worst_idx = idx
            return worst_idx, max_cr, hull
        except Exception:
            return -1, np.pi, None

    current_cr = best_radius
    
    T = 1e-3
    T_min = 1e-8
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        worst_idx, current_cr, hull = get_worst_facet(points)
        
        if worst_idx == -1:
            break
            
        if current_cr < best_radius:
            best_radius = current_cr
            best_points = points.copy()
            
        i, j, k = hull.simplices[worst_idx]
        vertex_to_move = rng.choice([i, j, k])
        
        p_i = points[i]
        p_j = points[j]
        p_k = points[k]
        c = p_i + p_j + p_k
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-10:
            c = points[vertex_to_move]
        else:
            c = c / norm_c
            
        step_size = 0.01
        noise = rng.normal(0, step_size * 0.5, 3)
        
        p_v = points[vertex_to_move]
        dir_vec = c - p_v
        dot = np.dot(dir_vec, p_v)
        dir_tangent = dir_vec - dot * p_v
        norm_dir = np.linalg.norm(dir_tangent)
        if norm_dir > 1e-10:
            dir_tangent = dir_tangent / norm_dir
        
        new_p_v = p_v + step_size * dir_tangent + noise
        new_p_v = new_p_v / np.linalg.norm(new_p_v)
        
        points[vertex_to_move] = new_p_v
        
        new_worst_idx, new_cr, _ = get_worst_facet(points)
        
        if new_cr < current_cr:
            current_cr = new_cr
        else:
            delta = new_cr - current_cr
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_cr = new_cr
                else:
                    points[vertex_to_move] = p_v
            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    final_radius = _compute_covering_radius(best_points)
    if final_radius < best_radius:
        best_radius = final_radius
        best_points = points.copy()

    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    print(f"Covering Radius: {_compute_covering_radius(pts) * 180 / np.pi:.6f} degrees")
