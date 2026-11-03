"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient.
    Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient for point i: sum_{j!=i} (x_i - x_j) / d_ij^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < 1e-18):
        return float('inf'), None
        
    dists = np.sqrt(dist_sq[mask])
    energy = -np.sum(np.log(dists))
    
    dist_sq_safe = dist_sq.copy()
    dist_sq_safe[dist_sq_safe == 0] = 1.0
    
    terms = diff / dist_sq_safe[:, :, np.newaxis]
    idx = np.arange(n)
    terms[idx, idx] = 0.0
    grad = np.sum(terms, axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate points using the Fibonacci spiral method."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    return _project_to_sphere(pts)

def _optimize_from_start(pts0, time_limit):
    """
    Optimize a single starting configuration using L-BFGS-B.
    Uses a wrapper that handles the sphere constraint via projection after each step,
    or more robustly, uses the fact that L-BFGS-B can handle bounds, but here we
    rely on the projection or use a penalty method? No, the standard trick for
    sphere constraints in L-BFGS-B is tricky.
    
    Better approach: Use Riemannian optimization or simply project after each L-BFGS step?
    L-BFGS-B does not support custom projections.
    
    Alternative: Use scipy.optimize.minimize with method='L-BFGS-B' but we need to handle
    the manifold. A common trick is to use spherical coordinates or just use a generic
    gradient descent with good line search, but the prompt suggests L-BFGS-B.
    
    Actually, a robust way to use L-BFGS-B on a sphere is to optimize in R^3 and add a
    penalty for deviation from the sphere, OR to use the fact that for N=100, we can
    just use a high-precision gradient descent with adaptive step size (like the current
    one but faster) or use `scipy.optimize.minimize` with `method='BFGS'` or `L-BFGS-B`
    on the tangent space?
    
    Let's stick to the most reliable method that beats simple GD:
    Use `scipy.optimize.minimize` with `method='L-BFGS-B'` is not directly applicable
    to the sphere constraint without a penalty.
    
    Instead, let's use a **Riemannian L-BFGS** implementation or a very robust
    **Gradient Descent with Nesterov momentum and adaptive step size** which is often
    competitive and easier to implement correctly for manifolds.
    
    However, the inspiration suggests L-BFGS-B. How to use it?
    One common trick: Parameterize the sphere using local coordinates? No, global is hard.
    
    Let's use **scipy.optimize.minimize** with **method='L-BFGS-B'** on the full R^3
    coordinates but with a **large penalty** for ||x||^2 - 1?
    This can be unstable.
    
    Alternative: Use **`scipy.optimize.minimize`** with **method='BFGS'** (quasi-Newton)
    which is often better than GD for smooth functions. But still, the constraint.
    
    Let's try a different approach: **Simulated Annealing** or **Basin Hopping**?
    Basin Hopping is great for finding global minima.
    
    Strategy:
    1. Generate multiple diverse starts (Fibonacci, Random).
    2. For each start, run a **Riemannian Gradient Descent** with **Momentum** and
       **Adaptive Step Size** (like the current code but potentially faster/more robust).
    3. Then, polish the result with **L-BFGS-B** using a **penalty method** for the
       sphere constraint? Or just a final very fine GD?
       
    Actually, the most effective "structural" change that is easy to implement and
    often outperforms simple GD is **Riemannian L-BFGS**. Since I can't easily import
    a Riemannian library, I will implement a **robust Gradient Descent with Momentum**
    and **Line Search** (Backtracking) which is a structural improvement over fixed-step
    GD.
    
    Wait, the current program *already* uses adaptive step size GD.
    
    Let's try **Basin Hopping** from SciPy. It combines local minimization with
    random perturbations, which is excellent for escaping local minima.
    """
    start_time = time.time()
    
    # Define the objective function for Basin Hopping
    # Basin Hopping requires a local minimizer.
    # We will use a simple Riemannian GD as the local minimizer for BH.
    
    def local_minimize(pts):
        """Local minimization using Riemannian GD with momentum."""
        pts = _project_to_sphere(pts.copy())
        current_energy, current_grad = _energy_and_grad(pts)
        if not np.isfinite(current_energy):
            return pts, current_energy
        
        best_pts = pts.copy()
        best_energy = current_energy
        
        step_size = 0.01
        momentum = 0.9
        v = np.zeros_like(pts)
        
        for _ in range(1000):
            if time.time() - start_time > time_limit:
                break
                
            # Update velocity
            v = momentum * v - step_size * current_grad
            
            # Move and project
            new_pts = pts + v
            new_pts = _project_to_sphere(new_pts)
            
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if not np.isfinite(new_energy):
                step_size *= 0.5
                v = 0.0
                continue
                
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
                
            # Simple line search / step adaptation
            if new_energy < current_energy:
                step_size = min(step_size * 1.1, 0.1)
            else:
                step_size *= 0.7
                
            pts = new_pts
            current_energy = new_energy
            current_grad = new_grad
            
            if np.linalg.norm(current_grad) < 1e-6:
                break
                
        return best_pts, best_energy
    
    # Run Basin Hopping
    # Note: Basin Hopping is stochastic. We set a seed for reproducibility in this run.
    # But we want diversity, so we might just run it once per start.
    
    # Actually, Basin Hopping can be slow. Let's use it for a few iterations.
    # For N=100, each local minimization takes ~1000 energy evals.
    # BH with 100 iterations -> 100k evals. 100k * 100^2 ops = 10^9 ops. ~1s in numpy.
    # So we can afford it.
    
    try:
        res = minimize(
            lambda x: _energy_and_grad(x.reshape(N, 3))[0],
            pts0.flatten(),
            method='L-BFGS-B',
            jac=lambda x: _energy_and_grad(x.reshape(N, 3))[1].flatten(),
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        # L-BFGS-B on R^3 does NOT respect the sphere constraint!
        # The result will be off the sphere. We must project.
        # But if it goes far off, the energy calculation in the wrapper might be wrong
        # if it doesn't handle the projection internally.
        # My wrapper `_energy_and_grad` expects points on the sphere (or close).
        # If L-BFGS-B moves points off the sphere, the energy is computed on those off-sphere points?
        # No, `_energy_and_grad` computes energy based on the input points.
        # If the points are not on the sphere, the energy is not the spherical energy.
        # So L-BFGS-B on R^3 is NOT valid for this problem unless we add a penalty.
        
        # Let's add a penalty term to the objective for L-BFGS-B.
        # E_total = E_sphere + lambda * sum(||x_i||^2 - 1)^2
        # This is a structural change: Penalty Method + L-BFGS-B.
        
    except Exception:
        pass

    # Given the complexity of making L-BFGS-B work correctly with the sphere constraint
    # (penalty tuning is fragile), let's stick to a **high-quality Riemannian Optimizer**.
    # A better structural change: **Conjugate Gradient** on the sphere?
    # Or **Adam** optimizer (adaptive learning rates per coordinate)?
    
    # Let's implement **Adam** on the sphere. It often converges faster than GD
    # and is robust to step size.
    
    def local_minimize_adam(pts):
        pts = _project_to_sphere(pts.copy())
        current_energy, current_grad = _energy_and_grad(pts)
        if not np.isfinite(current_energy):
            return pts, current_energy
        
        best_pts = pts.copy()
        best_energy = current_energy
        
        # Adam parameters
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m = np.zeros_like(pts)
        v = np.zeros_like(pts)
        t = 0
        
        for _ in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            t += 1
            m = beta1 * m + (1 - beta1) * current_grad
            v = beta2 * v + (1 - beta2) * (current_grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            update = m_hat / (np.sqrt(v_hat) + eps)
            
            new_pts = pts - lr * update
            new_pts = _project_to_sphere(new_pts)
            
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if not np.isfinite(new_energy):
                lr *= 0.5
                continue
                
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
                
            # Simple adaptive LR
            if new_energy < current_energy:
                lr = min(lr * 1.05, 0.1)
            else:
                lr *= 0.9
                
            pts = new_pts
            current_energy = new_energy
            current_grad = new_grad
            
            if np.linalg.norm(current_grad) < 1e-7:
                break
                
        return best_pts, best_energy

    return local_minimize_adam(pts0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit

    # Define initial configurations
    configs = []
    
    # 1. Fibonacci Spiral
    fib_pts = _fibonacci_sphere(N)
    configs.append(fib_pts)
    
    # 2. Random Uniform on Sphere (multiple seeds)
    for seed in [42, 1, 2, 3, 4, 5]:
        rand_pts = _random_sphere(N, seed)
        configs.append(rand_pts)
        
    best_pts_global = fib_pts
    best_energy_global = float('inf')
    
    # Optimize each configuration
    for idx, cfg in enumerate(configs):
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time
        remaining_configs = len(configs) - idx
        time_per_config = (deadline - time.time()) / remaining_configs
        time_per_config = min(time_per_config, 20.0)
        
        if time_per_config < 0.5:
            continue
            
        optimized_pts, energy = _optimize_from_start(cfg, time_limit=time_per_config)
        
        if energy < best_energy_global:
            best_energy_global = energy
            best_pts_global = optimized_pts
            
    # Final Polish with a very small step size GD
    if time.time() < deadline - 2.0:
        # Use the same Adam optimizer but with a smaller learning rate?
        # Or just run it again on the best point.
        final_pts, final_energy = _optimize_from_start(best_pts_global, time_limit=5.0)
        if final_energy < best_energy_global:
            best_pts_global = final_pts
            best_energy_global = final_energy

    # Ensure output is valid
    best_pts_global = _project_to_sphere(best_pts_global)
    
    return best_pts_global
# EVOLVE-BLOCK-END
